"""Precompute per-recording robust stats from the window cache (no EDF rebuild).

Writes data/artifacts/window_cache/record_stats.npz with median/IQR per channel
for each record_id, usable for patient/recording-level normalization at train time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from seizure_detector.cache import WINDOW_CACHE_DIR, load_window_cache
from seizure_detector.dataset import load_windows

OUT = WINDOW_CACHE_DIR / "record_stats.npz"
MAX_WINDOWS_PER_RECORD = 64


def main() -> None:
    windows = load_windows()
    x, mask, meta = load_window_cache(WINDOW_CACHE_DIR)
    # Sample windows per record for robust location/scale.
    medians = {}
    iqrs = {}
    counts = {}
    rng = np.random.default_rng(1337)
    grouped = windows.groupby("record_id", sort=False)
    n_groups = windows["record_id"].nunique()
    for i, (rid, group) in enumerate(grouped):
        ids = group["window_id"].to_numpy()
        if len(ids) > MAX_WINDOWS_PER_RECORD:
            ids = rng.choice(ids, size=MAX_WINDOWS_PER_RECORD, replace=False)
        sample = np.asarray(x[ids], dtype=np.float32)  # [N, C, T]
        # Pool over time then robust stats over windows.
        flat = sample.reshape(sample.shape[0], sample.shape[1], -1)
        # median over time then over windows
        per_win = np.median(flat, axis=-1)  # [N, C]
        med = np.median(per_win, axis=0)
        q75 = np.percentile(per_win, 75, axis=0)
        q25 = np.percentile(per_win, 25, axis=0)
        iqr = np.maximum(q75 - q25, 1e-3)
        medians[rid] = med.astype(np.float32)
        iqrs[rid] = iqr.astype(np.float32)
        counts[rid] = int(len(ids))
        if (i + 1) % 500 == 0:
            print(f"records {i+1}/{n_groups}", flush=True)
    rids = list(medians.keys())
    np.savez_compressed(
        OUT,
        record_ids=np.asarray(rids, dtype=object),
        median=np.stack([medians[r] for r in rids]),
        iqr=np.stack([iqrs[r] for r in rids]),
        n_windows=np.asarray([counts[r] for r in rids], dtype=np.int32),
    )
    (WINDOW_CACHE_DIR / "record_stats_meta.json").write_text(
        json.dumps({"n_records": len(rids), "max_windows_per_record": MAX_WINDOWS_PER_RECORD}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUT} for {len(rids)} records")


if __name__ == "__main__":
    main()
