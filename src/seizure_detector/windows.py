"""Build a compact weak-label window manifest without materializing signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, SPLITS_PATH, WINDOWS_PATH, WindowConfig
from .labels import read_seizure_times
from .paths import load_splits, recording_paths
from .preprocess import recording_info


def _window_rows(
    rec,
    paths: dict[str, Path],
    cfg: WindowConfig,
    rng: np.random.Generator,
) -> list[dict]:
    try:
        info = recording_info(paths["edf"], file_size_bytes=getattr(rec, "size_bytes", None))
    except (ValueError, OSError):
        return []
    duration = float(info["duration_seconds"])
    if duration < cfg.window_seconds or not paths["xltek"].is_file():
        return []
    events = read_seizure_times(paths["xltek"], info["start"])
    events = events[(events >= 0) & (events <= duration)]
    starts = np.arange(0, duration - cfg.window_seconds + 1, cfg.stride_seconds)
    centers = starts + cfg.window_seconds / 2
    if len(events):
        distance = np.min(np.abs(centers[:, None] - events[None, :]), axis=1)
        positive = distance <= cfg.positive_radius_seconds
        negative = distance >= cfg.negative_guard_seconds
    else:
        positive = np.zeros(len(starts), dtype=bool)
        negative = np.ones(len(starts), dtype=bool)

    positive_idx = np.flatnonzero(positive)
    negative_idx = np.flatnonzero(negative)
    if len(negative_idx) > cfg.max_negative_windows_per_recording:
        negative_idx = rng.choice(
            negative_idx, cfg.max_negative_windows_per_recording, replace=False
        )
    chosen = np.concatenate([positive_idx, negative_idx])
    labels = np.concatenate(
        [np.ones(len(positive_idx), dtype=int), np.zeros(len(negative_idx), dtype=int)]
    )
    rows = []
    for index, label in zip(chosen, labels, strict=True):
        rows.append(
            {
                "record_id": rec.record_id,
                "patient": int(rec.patient),
                "session": int(rec.session),
                "split": rec.split,
                "edf_path": str(paths["edf"]),
                "start_seconds": float(starts[index]),
                "duration_seconds": cfg.window_seconds,
                "label": int(label),
                "label_source": "xltek_point_marker",
                "n_recording_seizure_markers": int(len(events)),
            }
        )
    return rows


def build_window_manifest(
    splits_path: Path = SPLITS_PATH,
    output_path: Path = WINDOWS_PATH,
    *,
    raw_dir: Path = RAW_DIR,
    config: WindowConfig = WindowConfig(),
    include_splits: tuple[str, ...] = ("train", "val", "test"),
    max_records: int | None = None,
    require_full_edf: bool = True,
) -> pd.DataFrame:
    """Build weak-label window metadata.

    When ``require_full_edf=False`` (Phase A of full-scale ingestion), only a
    tiny EDF header stub (~8KB) needs to exist locally per recording so
    ``recording_info`` can read duration and sample rate without the full signal.
    """
    splits = load_splits(splits_path)
    splits = splits[splits["split"].isin(include_splits)]
    if max_records is not None:
        splits = splits.head(max_records)
    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []
    skipped = 0
    for rec in splits.itertuples(index=False):
        paths = recording_paths(rec.record_id, raw_dir)
        if require_full_edf:
            if not paths["edf"].is_file():
                skipped += 1
                continue
        else:
            if not paths["edf"].is_file() or paths["edf"].stat().st_size < 256:
                skipped += 1
                continue
        rows.extend(_window_rows(rec, paths, config, rng))
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise RuntimeError("No windows created. Audit EDF and annotation paths first.")
    if windows.groupby("patient")["split"].nunique().gt(1).any():
        raise RuntimeError("Patient leakage detected while creating windows")
    # Stable join key for the precomputed array cache (seizure_detector.cache);
    # must stay put once assigned so cached rows keep matching these windows.
    windows["window_id"] = np.arange(len(windows), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(output_path, index=False)
    summary = (
        windows.groupby("split")
        .agg(windows=("label", "size"), positives=("label", "sum"), patients=("patient", "nunique"))
    )
    print(summary.to_string())
    if skipped:
        print(f"Skipped {skipped} selected records without local EDFs.")
    print(f"Wrote {len(windows):,} windows to {output_path}")
    return windows
