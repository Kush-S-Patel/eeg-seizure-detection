"""Precompute filtered, resampled window arrays so training reads plain tensors."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SignalConfig, WINDOW_CACHE_DIR
from .preprocess import (
    MONTAGE_NAMES,
    build_bipolar_montage,
    filter_and_resample,
    open_raw,
    resolve_montage_channels,
    robust_scale,
)

_KEY_COLUMNS = ("window_id", "record_id", "start_seconds", "duration_seconds")
_MAX_CHUNK_BYTES = 200_000_000
_PROGRESS_NAME = "completed_records.json"


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def _progress_path(cache_dir: Path) -> Path:
    return cache_dir / _PROGRESS_NAME


def _fingerprint(windows: pd.DataFrame) -> str:
    key = windows[list(_KEY_COLUMNS)].sort_values("window_id")
    digest = hashlib.sha256(pd.util.hash_pandas_object(key, index=False).values.tobytes())
    return digest.hexdigest()


def load_progress(cache_dir: Path) -> set[str]:
    path = _progress_path(cache_dir)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(payload.get("completed", []))
    except (OSError, json.JSONDecodeError):
        return set()


def save_progress(cache_dir: Path, completed: set[str]) -> None:
    _progress_path(cache_dir).write_text(
        json.dumps({"completed": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


def cache_is_valid(cache_dir: Path, windows: pd.DataFrame, signal_config: SignalConfig) -> bool:
    meta_path = _meta_path(cache_dir)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("signal_config") != signal_config.__dict__:
        return False
    if meta.get("fingerprint") != _fingerprint(windows):
        return False
    completed = load_progress(cache_dir)
    expected = set(windows["record_id"].unique())
    return expected.issubset(completed)


def cache_covers(cache_dir: Path, windows: pd.DataFrame, signal_config: SignalConfig) -> bool:
    """True if an existing cache can serve ``windows`` (subset OK; fingerprint need not match)."""
    meta_path = _meta_path(cache_dir)
    if not meta_path.exists() or "window_id" not in windows.columns or windows.empty:
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("signal_config") != signal_config.__dict__:
        return False
    if int(windows["window_id"].max()) >= meta.get("n_rows", 0):
        return False
    x_path = Path(cache_dir) / "x.f32"
    expected = int(meta["n_rows"]) * int(meta["channels"]) * int(meta["samples"]) * 4
    if not x_path.exists() or x_path.stat().st_size != expected:
        return False
    completed = load_progress(cache_dir)
    return set(windows["record_id"].unique()).issubset(completed)


def load_window_cache(cache_dir: Path = WINDOW_CACHE_DIR):
    meta = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
    shape_x = (meta["n_rows"], meta["channels"], meta["samples"])
    shape_mask = (meta["n_rows"], meta["channels"])
    x = np.memmap(cache_dir / "x.f32", dtype=np.float32, mode="r", shape=shape_x)
    mask = np.memmap(cache_dir / "mask.f32", dtype=np.float32, mode="r", shape=shape_mask)
    return x, mask, meta


def ensure_window_cache(
    windows: pd.DataFrame,
    signal_config: SignalConfig = SignalConfig(),
    cache_dir: Path = WINDOW_CACHE_DIR,
    *,
    rebuild: bool = False,
    workers: int = 1,
) -> Path:
    """Reuse a covering cache when possible; never rebuild from a subset fingerprint.

    Training on forecast / filtered window tables must not re-fingerprint or
    reallocate the full ``x.f32`` store — that truncates ~250GB to zeros.
    """
    cache_dir = Path(cache_dir)
    if not rebuild and cache_covers(cache_dir, windows, signal_config):
        print(f"Reusing valid window cache at {cache_dir}", flush=True)
        return cache_dir
    if not rebuild and cache_is_valid(cache_dir, windows, signal_config):
        print(f"Reusing valid window cache at {cache_dir}", flush=True)
        return cache_dir
    # Building requires the caller's table to define allocation size. Refuse if an
    # existing larger cache already covers these window_ids (subset mismatch).
    meta_path = _meta_path(cache_dir)
    if meta_path.exists() and not rebuild and "window_id" in windows.columns and not windows.empty:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            max_id = int(windows["window_id"].max())
            if max_id < int(meta.get("n_rows", 0)):
                x_path = cache_dir / "x.f32"
                expected = int(meta["n_rows"]) * int(meta["channels"]) * int(meta["samples"]) * 4
                if x_path.exists() and x_path.stat().st_size == expected:
                    completed = load_progress(cache_dir)
                    missing = set(windows["record_id"].astype(str).unique()) - completed
                    raise RuntimeError(
                        f"Window cache at {cache_dir} is sized for a larger table "
                        f"(n_rows={meta.get('n_rows')}) but does not cover this subset "
                        f"({len(missing)} recordings missing from completed_records.json). "
                        "Refusing to reallocate (would wipe x.f32). "
                        "Re-run Phase B fill for the missing recordings, or pass "
                        "rebuild_cache only with the FULL windows.parquet."
                    )
        except RuntimeError:
            raise
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return build_window_cache(
        windows, signal_config, cache_dir, force=rebuild, workers=workers
    )


def _process_chunk(
    raw,
    picks: list[int],
    present: list[str],
    read_start: int,
    read_stop: int,
    rows: list,
    source_rate: float,
    signal_config: SignalConfig,
    samples: int,
    ratio: float,
    x: np.memmap,
    mask: np.memmap,
) -> None:
    read_start = max(0, int(read_start))
    read_stop = min(raw.n_times, int(read_stop))
    values = raw.get_data(picks=picks, start=read_start, stop=read_stop)
    by_name = dict(zip(present, values, strict=True))
    montage, channel_mask = build_bipolar_montage(by_name, expected_len=read_stop - read_start)
    processed = filter_and_resample(montage, source_rate, signal_config)
    for row in rows:
        local_start = int(round((row.start_sample - read_start) * ratio))
        window = processed[:, local_start : local_start + samples]
        if window.shape[-1] < samples:
            window = np.pad(window, ((0, 0), (0, samples - window.shape[-1])))
        x[row.window_id] = robust_scale(window, signal_config.clip)
        mask[row.window_id] = channel_mask


def _write_zero_windows(
    group: pd.DataFrame,
    samples: int,
    x: np.memmap,
    mask: np.memmap,
) -> None:
    """Placeholder rows when a recording has no usable montage channels."""
    zeros = np.zeros((len(MONTAGE_NAMES), samples), dtype=np.float32)
    zero_mask = np.zeros(len(MONTAGE_NAMES), dtype=np.float32)
    for row in group.itertuples(index=False):
        x[row.window_id] = zeros
        mask[row.window_id] = zero_mask


def _fill_group(
    edf_path: str,
    group: pd.DataFrame,
    signal_config: SignalConfig,
    samples: int,
    x: np.memmap,
    mask: np.memmap,
) -> None:
    raw = open_raw(edf_path)
    source_rate = float(raw.info["sfreq"])
    channel_index, present = resolve_montage_channels(raw.ch_names)
    if not present:
        print(
            f"    ! no montage channels in {edf_path}, writing zero windows",
            flush=True,
        )
        _write_zero_windows(group, samples, x, mask)
        return
    picks = [channel_index[name] for name in present]
    ratio = signal_config.sample_rate / source_rate
    bytes_per_sample = max(len(picks), 1) * 8

    frame = group.copy()
    frame["start_sample"] = np.round(frame["start_seconds"].to_numpy() * source_rate).astype(int)
    frame["stop_sample"] = frame["start_sample"] + np.round(
        frame["duration_seconds"].to_numpy() * source_rate
    ).astype(int)
    frame = frame.sort_values("start_sample")

    chunk_rows: list = []
    chunk_start: int | None = None
    chunk_stop: int | None = None
    for row in frame.itertuples(index=False):
        start = min(chunk_start, row.start_sample) if chunk_start is not None else row.start_sample
        stop = max(chunk_stop, row.stop_sample) if chunk_stop is not None else row.stop_sample
        if chunk_rows and (stop - start) * bytes_per_sample > _MAX_CHUNK_BYTES:
            _process_chunk(
                raw, picks, present, chunk_start, chunk_stop, chunk_rows,
                source_rate, signal_config, samples, ratio, x, mask,
            )
            chunk_rows, chunk_start, chunk_stop = [row], row.start_sample, row.stop_sample
        else:
            chunk_rows.append(row)
            chunk_start, chunk_stop = start, stop
    if chunk_rows:
        _process_chunk(
            raw, picks, present, chunk_start, chunk_stop, chunk_rows,
            source_rate, signal_config, samples, ratio, x, mask,
        )


def _fill_group_worker(args: tuple) -> str:
    edf_path, group_dict, signal_dict, samples, cache_dir_str = args
    signal_config = SignalConfig(**signal_dict)
    group = pd.DataFrame(group_dict)
    cache_dir = Path(cache_dir_str)
    meta = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
    shape_x = (meta["n_rows"], meta["channels"], meta["samples"])
    shape_mask = (meta["n_rows"], meta["channels"])
    x = np.memmap(cache_dir / "x.f32", dtype=np.float32, mode="r+", shape=shape_x)
    mask = np.memmap(cache_dir / "mask.f32", dtype=np.float32, mode="r+", shape=shape_mask)
    _fill_group(edf_path, group, signal_config, samples, x, mask)
    x.flush()
    mask.flush()
    return str(group["record_id"].iloc[0])


def allocate_cache(
    windows: pd.DataFrame,
    signal_config: SignalConfig = SignalConfig(),
    cache_dir: Path = WINDOW_CACHE_DIR,
    *,
    force: bool = False,
) -> Path:
    """Create empty memmaps sized for the full window table (Phase A)."""
    if "window_id" not in windows.columns:
        raise ValueError("windows table is missing `window_id`")
    if windows["duration_seconds"].nunique() != 1:
        raise ValueError("All windows must share the same duration_seconds")

    cache_dir = Path(cache_dir)
    if not force and _meta_path(cache_dir).exists():
        meta = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
        if (
            meta.get("signal_config") == signal_config.__dict__
            and meta.get("fingerprint") == _fingerprint(windows)
        ):
            print(f"Reusing allocated cache at {cache_dir}", flush=True)
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    n_rows = int(windows["window_id"].max()) + 1
    window_seconds = float(windows["duration_seconds"].iloc[0])
    samples = int(round(window_seconds * signal_config.sample_rate))
    channels = len(MONTAGE_NAMES)

    x_path = cache_dir / "x.f32"
    mask_path = cache_dir / "mask.f32"
    shape_x = (n_rows, channels, samples)
    shape_mask = (n_rows, channels)
    expected_x_bytes = int(np.prod(shape_x)) * 4
    expected_mask_bytes = int(np.prod(shape_mask)) * 4

    # Size the files only — do not zero-fill ~250GB; Phase B writes each row in place.
    # CRITICAL: mode "w+" truncates. Never open w+ when an existing file has a
    # different size (e.g. training on a subset whose max window_id is smaller).
    for path, expected, shape in (
        (x_path, expected_x_bytes, shape_x),
        (mask_path, expected_mask_bytes, shape_mask),
    ):
        if path.exists():
            existing = path.stat().st_size
            if existing == expected:
                continue
            if existing > expected and not force:
                raise RuntimeError(
                    f"Refusing to shrink {path} from {existing:,} to {expected:,} bytes "
                    f"(would wipe cached windows). Pass force=True only when intentionally "
                    f"rebuilding, and prefer allocate_cache on the FULL windows table."
                )
            if existing < expected and not force:
                raise RuntimeError(
                    f"Refusing to recreate {path} ({existing:,} -> {expected:,} bytes) "
                    f"without force=True; recreating with w+ would wipe existing cache data. "
                    f"Grow the file explicitly or re-run Phase A allocate on a clean path."
                )
            if force:
                print(
                    f"  ! force-reallocating {path.name} ({existing:,} -> {expected:,} bytes)",
                    flush=True,
                )
                np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
        else:
            np.memmap(path, dtype=np.float32, mode="w+", shape=shape)

    _meta_path(cache_dir).write_text(
        json.dumps(
            {
                "n_rows": n_rows,
                "channels": channels,
                "samples": samples,
                "window_seconds": window_seconds,
                "signal_config": signal_config.__dict__,
                "fingerprint": _fingerprint(windows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if force or not _progress_path(cache_dir).exists():
        save_progress(cache_dir, set())
    print(
        f"Allocated cache for {n_rows:,} windows ({n_rows * channels * samples * 4 / 1024**3:.1f} GB x arrays)",
        flush=True,
    )
    return cache_dir


def fill_cache_batch(
    batch_windows: pd.DataFrame,
    signal_config: SignalConfig = SignalConfig(),
    cache_dir: Path = WINDOW_CACHE_DIR,
    *,
    workers: int = 1,
) -> int:
    """Fill cache rows for one batch of recordings. Returns recordings processed."""
    if not _meta_path(cache_dir).exists():
        raise FileNotFoundError(f"Cache not allocated at {cache_dir}; run allocate_cache first")

    meta = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
    samples = int(meta["samples"])
    completed = load_progress(cache_dir)
    groups = [
        (edf_path, group)
        for edf_path, group in batch_windows.groupby("edf_path")
        if str(group["record_id"].iloc[0]) not in completed
    ]
    if not groups:
        return 0

    signal_dict = signal_config.__dict__
    cache_dir_str = str(cache_dir)
    if workers <= 1:
        shape_x = (meta["n_rows"], meta["channels"], meta["samples"])
        shape_mask = (meta["n_rows"], meta["channels"])
        x = np.memmap(cache_dir / "x.f32", dtype=np.float32, mode="r+", shape=shape_x)
        mask = np.memmap(cache_dir / "mask.f32", dtype=np.float32, mode="r+", shape=shape_mask)
        filled = 0
        for edf_path, group in groups:
            record_id = str(group["record_id"].iloc[0])
            try:
                _fill_group(edf_path, group, signal_config, samples, x, mask)
            except Exception as exc:
                print(f"    ! cache fill skipped for {edf_path}: {exc}", flush=True)
                continue
            completed.add(record_id)
            filled += 1
            if filled % 5 == 0:
                save_progress(cache_dir, completed)
        x.flush()
        mask.flush()
    else:
        tasks = [
            (edf_path, group.to_dict(orient="list"), signal_dict, samples, cache_dir_str)
            for edf_path, group in groups
        ]
        filled = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fill_group_worker, task): task[0] for task in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                edf_path = futures[fut]
                try:
                    record_id = fut.result()
                except Exception as exc:
                    print(f"    ! cache fill skipped for {edf_path}: {exc}", flush=True)
                    continue
                completed.add(record_id)
                filled += 1
                if i % 5 == 0 or i == len(futures):
                    print(f"  cached {filled}/{len(futures)} recordings in batch", flush=True)
                    save_progress(cache_dir, completed)

    save_progress(cache_dir, completed)
    return filled


def build_window_cache(
    windows: pd.DataFrame,
    signal_config: SignalConfig = SignalConfig(),
    cache_dir: Path = WINDOW_CACHE_DIR,
    *,
    force: bool = False,
    workers: int = 1,
) -> Path:
    """Materialize every window (single-machine path; full-scale uses batches)."""
    if not force and cache_covers(cache_dir, windows, signal_config):
        print(f"Reusing valid window cache at {cache_dir}", flush=True)
        return cache_dir
    if not force and cache_is_valid(cache_dir, windows, signal_config):
        print(f"Reusing valid window cache at {cache_dir}", flush=True)
        return cache_dir

    allocate_cache(windows, signal_config, cache_dir, force=force)
    pending = windows[~windows["record_id"].isin(load_progress(cache_dir))]
    if len(pending):
        fill_cache_batch(pending, signal_config, cache_dir, workers=workers)
    print(f"Cached {len(windows):,} windows to {cache_dir}", flush=True)
    return cache_dir
