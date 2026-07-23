"""Full-scale (10TB) rolling ingestion: header manifest -> batched cache -> delete raw.

Phase A (cheap, ~minutes):
  1. Fetch ~8KB EDF header stubs for every annotated recording.
  2. Build the complete ``windows.parquet`` (~3M windows, all patients).
  3. Pre-allocate the window-array cache memmaps.

Phase B (heavy, resumable, ~hours):
  For each ~300GB batch of recordings:
    download full EDFs -> fill cache rows -> delete raw EDFs (+ json/channels).
  Annotations (Xltek CSVs) are kept permanently (already synced in metadata stage).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seizure_detector.cache import allocate_cache, fill_cache_batch, load_progress
from seizure_detector.config import SignalConfig, WINDOW_CACHE_DIR, WINDOWS_PATH, WindowConfig
from seizure_detector.windows import build_window_manifest

from . import awscli, config, download
from .config import DownloadConfig


def _edf_local_path(record_id: str) -> Path:
    return config.RAW_DIR / (record_id + config.EDF_SUFFIX)


def _records_with_complete_edf(batch: pd.DataFrame) -> pd.DataFrame:
    """Recordings whose local EDF matches the manifest size (ready for cache fill)."""
    ready: list[dict] = []
    for row in batch.itertuples(index=False):
        edf = _edf_local_path(row.record_id)
        if edf.is_file() and edf.stat().st_size == int(row.size_bytes):
            ready.append(row._asdict())
    if not ready:
        return batch.iloc[0:0].copy()
    return pd.DataFrame(ready)


def _sidecar_paths(record_id: str) -> list[Path]:
    base = config.RAW_DIR / record_id
    return [
        Path(f"{base}_eeg.json"),
        Path(f"{base}_channels.tsv"),
    ]


def download_phase_a_metadata(
    selection: pd.DataFrame,
    *,
    max_workers: int = 32,
) -> tuple[int, int]:
    """Fetch BIDS ``*_eeg.json`` sidecars and 256-byte EDF prefixes for Phase A."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config.ensure_dirs()
    tasks = []
    for row in selection.itertuples(index=False):
        record_id = row.record_id
        json_local = config.RAW_DIR / (record_id + config.SIDECAR_SUFFIXES["json"])
        edf_local = _edf_local_path(record_id)
        if json_local.is_file() and edf_local.is_file() and edf_local.stat().st_size >= 184:
            continue
        tasks.append((record_id, json_local, edf_local))

    if not tasks:
        print("  all Phase A metadata sidecars already present", flush=True)
        return len(selection), 0

    print(f"  fetching metadata for {len(tasks):,} recordings (json + EDF prefix)...", flush=True)
    ok = failed = 0

    def _fetch_one(record_id: str, json_local: Path, edf_local: Path) -> bool:
        json_key = record_id + config.SIDECAR_SUFFIXES["json"]
        edf_key = record_id + config.EDF_SUFFIX
        json_ok = awscli.cp(json_key, json_local, warn_on_missing=False)
        edf_ok = awscli.get_range(edf_key, edf_local, end_byte=255)
        return json_ok and edf_ok

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, record_id, json_local, edf_local): record_id
            for record_id, json_local, edf_local in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                failed += 1
            if i % 500 == 0 or i == len(futures):
                print(f"  metadata progress: {i}/{len(futures)} (ok={ok}, failed={failed})", flush=True)
    return ok, failed


def download_header_stubs(
    selection: pd.DataFrame,
    *,
    max_workers: int = 32,
) -> tuple[int, int]:
    """Backward-compatible alias for Phase A metadata download."""
    return download_phase_a_metadata(selection, max_workers=max_workers)


def phase_a_manifest_and_allocate(
    *,
    window_config: WindowConfig | None = None,
    signal_config: SignalConfig | None = None,
    cache_dir: Path | None = None,
    header_workers: int = 32,
) -> pd.DataFrame:
    """Phase A: header stubs -> full windows.parquet -> allocate cache."""
    from . import artifacts

    window_config = window_config or WindowConfig()
    signal_config = signal_config or SignalConfig()
    cache_dir = cache_dir or WINDOW_CACHE_DIR

    splits = artifacts.load_df(config.SPLIT_PATH)
    print("[fullscale/A] downloading BIDS metadata sidecars", flush=True)
    download_phase_a_metadata(splits, max_workers=header_workers)

    print("[fullscale/A] building complete window manifest from headers", flush=True)
    windows = build_window_manifest(
        splits_path=config.SPLIT_PATH,
        output_path=WINDOWS_PATH,
        raw_dir=config.RAW_DIR,
        config=window_config,
        require_full_edf=False,
    )

    print("[fullscale/A] pre-allocating window cache", flush=True)
    allocate_cache(windows, signal_config, cache_dir)
    return windows


def _batch_chunks(selection: pd.DataFrame, batch_gb: float) -> list[pd.DataFrame]:
    """Split recordings into byte-budget batches."""
    batch_bytes = int(batch_gb * 1024**3)
    chunks: list[pd.DataFrame] = []
    current_rows: list[dict] = []
    current_bytes = 0
    for row in selection.sort_values(["patient", "session"]).itertuples(index=False):
        size = int(row.size_bytes)
        if current_rows and current_bytes + size > batch_bytes:
            chunks.append(pd.DataFrame(current_rows))
            current_rows, current_bytes = [], 0
        current_rows.append(row._asdict())
        current_bytes += size
    if current_rows:
        chunks.append(pd.DataFrame(current_rows))
    return chunks


def _delete_raw_batch(batch: pd.DataFrame) -> None:
    """Remove downloaded EDFs and non-annotation sidecars for one batch."""
    for row in batch.itertuples(index=False):
        edf = _edf_local_path(row.record_id)
        if edf.is_file():
            edf.unlink()
        for path in _sidecar_paths(row.record_id):
            if path.is_file():
                path.unlink()


def _download_until_ready(batch: pd.DataFrame, dl: DownloadConfig) -> pd.DataFrame:
    """Download until every recording in the batch has a complete EDF on disk."""
    while True:
        ready = _records_with_complete_edf(batch)
        missing = batch[~batch["record_id"].isin(ready["record_id"])]
        if missing.empty:
            return ready
        print(
            f"  downloading {len(missing):,} recordings "
            f"({len(ready):,} already complete)...",
            flush=True,
        )
        before = len(ready)
        download.download_selection(missing, dl=dl)
        ready = _records_with_complete_edf(batch)
        if len(ready) <= before:
            print(
                f"  warning: {len(batch) - len(ready):,} recordings still missing EDF "
                "(continuing with those available)",
                flush=True,
            )
            return ready


def phase_b_rolling_cache(
    windows: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    batch_gb: float = 300.0,
    signal_config: SignalConfig | None = None,
    cache_dir: Path | None = None,
    download_config: DownloadConfig | None = None,
    cache_workers: int | None = None,
) -> None:
    """Phase B: batched download -> cache fill -> delete raw (resumable)."""
    signal_config = signal_config or SignalConfig()
    cache_dir = cache_dir or WINDOW_CACHE_DIR
    download_config = download_config or config.DEFAULT_DOWNLOAD
    if cache_workers is None:
        import os

        cache_workers = max(1, (os.cpu_count() or 4) - 2)

    completed = load_progress(cache_dir)
    pending = selection[~selection["record_id"].isin(completed)]
    if pending.empty:
        print("[fullscale/B] all recordings already cached", flush=True)
        return

    chunks = _batch_chunks(pending, batch_gb)
    print(
        f"[fullscale/B] {len(pending):,} recordings pending in {len(chunks)} batches "
        f"(~{batch_gb:.0f} GB each, cache_workers={cache_workers})",
        flush=True,
    )

    for batch_idx, batch in enumerate(chunks, 1):
        batch_ids = set(batch["record_id"])
        batch_windows = windows[windows["record_id"].isin(batch_ids)]
        batch_gb_actual = batch["size_bytes"].sum() / 1024**3
        print(
            f"\n[fullscale/B] batch {batch_idx}/{len(chunks)}: "
            f"{len(batch):,} recordings, {batch_gb_actual:.1f} GB",
            flush=True,
        )

        ready = _download_until_ready(batch, download_config)
        ready_ids = set(ready["record_id"])
        batch_windows = batch_windows[batch_windows["record_id"].isin(ready_ids)]
        if batch_windows.empty:
            print("  no recordings ready for cache fill in this batch", flush=True)
            continue
        filled = fill_cache_batch(
            batch_windows, signal_config, cache_dir, workers=cache_workers
        )
        print(f"  filled cache for {filled} recordings", flush=True)
        _delete_raw_batch(ready)
        print("  deleted raw EDFs for batch", flush=True)


def run_fullscale(
    *,
    skip_phase_a: bool = False,
    batch_gb: float = 300.0,
    window_config: WindowConfig | None = None,
    signal_config: SignalConfig | None = None,
    cache_dir: Path | None = None,
    download_config: DownloadConfig | None = None,
    header_workers: int = 32,
    cache_workers: int | None = None,
) -> None:
    """End-to-end full-scale ingestion after manifest/index/select/split."""
    from . import artifacts

    signal_config = signal_config or SignalConfig()
    cache_dir = cache_dir or WINDOW_CACHE_DIR
    selection = artifacts.load_df(config.SPLIT_PATH)

    if skip_phase_a and WINDOWS_PATH.exists():
        windows = pd.read_parquet(WINDOWS_PATH)
        print(f"[fullscale] reusing existing windows.parquet ({len(windows):,} rows)", flush=True)
        allocate_cache(windows, signal_config, cache_dir)
    else:
        windows = phase_a_manifest_and_allocate(
            window_config=window_config,
            signal_config=signal_config,
            cache_dir=cache_dir,
            header_workers=header_workers,
        )

    phase_b_rolling_cache(
        windows,
        selection,
        batch_gb=batch_gb,
        signal_config=signal_config,
        cache_dir=cache_dir,
        download_config=download_config,
        cache_workers=cache_workers,
    )
    completed = load_progress(cache_dir)
    print(
        f"\n[fullscale] complete: {len(completed):,}/{selection['record_id'].nunique():,} "
        f"recordings cached at {cache_dir}",
        flush=True,
    )


def cleanup_header_stubs(selection: pd.DataFrame) -> None:
    """Remove tiny header-only EDF stubs after Phase B (optional housekeeping)."""
    removed = 0
    for row in selection.itertuples(index=False):
        edf = _edf_local_path(row.record_id)
        if edf.is_file() and edf.stat().st_size <= 8192:
            edf.unlink()
            removed += 1
    if removed:
        print(f"  removed {removed:,} header stubs", flush=True)
