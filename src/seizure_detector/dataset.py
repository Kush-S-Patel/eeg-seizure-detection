"""PyTorch dataset that reads precomputed window arrays from the fast cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .cache import WINDOW_CACHE_DIR, cache_covers, load_window_cache
from .config import SignalConfig, WINDOWS_PATH
from .preprocess import extract_window


class EEGWindowDataset(Dataset):
    """Reads cached float32 window arrays when a valid cache is available.

    Falls back to on-demand EDF extraction (slow: disk I/O plus filtering per
    sample) only when no cache has been built yet. Training/evaluation should
    always run `seizure-detector cache` (or `train`, which builds it
    automatically) first, since the fallback path re-filters every window on
    every access.
    """

    def __init__(
        self,
        windows: pd.DataFrame,
        signal_config: SignalConfig = SignalConfig(),
        cache_dir: Path | None = WINDOW_CACHE_DIR,
        *,
        record_norm: bool = False,
    ):
        self.windows = windows.reset_index(drop=True)
        self.signal_config = signal_config
        self._cache = None
        self._record_median: dict[str, np.ndarray] = {}
        self._record_iqr: dict[str, np.ndarray] = {}
        if cache_dir is not None and cache_covers(cache_dir, windows, signal_config):
            self._cache = load_window_cache(cache_dir)
        elif cache_dir is not None:
            print(
                f"  ! no valid window cache at {cache_dir}; falling back to slow "
                "per-window EDF extraction. Run `seizure-detector cache` first."
            )
        if record_norm and cache_dir is not None:
            stats_path = Path(cache_dir) / "record_stats.npz"
            if stats_path.exists():
                payload = np.load(stats_path, allow_pickle=True)
                for rid, med, iqr in zip(
                    payload["record_ids"], payload["median"], payload["iqr"]
                ):
                    self._record_median[str(rid)] = med.astype(np.float32)
                    self._record_iqr[str(rid)] = np.maximum(iqr.astype(np.float32), 1e-3)
            else:
                print(
                    f"  ! record_norm requested but {stats_path} missing; "
                    "run scripts/compute_record_stats.py first."
                )

    def __len__(self) -> int:
        return len(self.windows)

    def _read(self, row) -> tuple:
        if self._cache is not None:
            x, mask, _meta = self._cache
            window_id = int(row.window_id)
            # np.array() copies out of the read-only memmap so the resulting
            # tensor is writable (avoids a PyTorch non-writable-array warning).
            data = np.array(x[window_id])
            m = np.array(mask[window_id])
        else:
            data, m = extract_window(
                Path(row.edf_path),
                float(row.start_seconds),
                float(row.duration_seconds),
                self.signal_config,
            )
        rid = str(row.record_id)
        if rid in self._record_median:
            med = self._record_median[rid][:, None]
            iqr = self._record_iqr[rid][:, None]
            data = ((data - med) / iqr).astype(np.float32)
            data = np.clip(data, -8.0, 8.0)
        return data, m

    def __getitem__(self, index: int) -> dict:
        row = self.windows.iloc[index]
        data, mask = self._read(row)
        return {
            "x": torch.from_numpy(data),
            "channel_mask": torch.from_numpy(mask),
            "y": torch.tensor(float(row.label), dtype=torch.float32),
            "index": torch.tensor(index),
        }


def load_windows(path: Path = WINDOWS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `seizure-detector prepare`.")
    windows = pd.read_parquet(path)
    required = {
        "edf_path",
        "start_seconds",
        "duration_seconds",
        "label",
        "patient",
        "split",
        "window_id",
    }
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(
            f"Window table missing columns: {sorted(missing)}. "
            "Re-run `seizure-detector prepare` to regenerate it."
        )
    return windows


def balanced_sampler(labels: pd.Series, seed: int = 1337) -> WeightedRandomSampler | None:
    counts = labels.value_counts()
    if len(counts) < 2 or (counts == 0).any():
        return None
    weights = labels.map(
        {label: 1.0 / count for label, count in counts.items()}
    ).to_numpy(copy=True)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def limit_windows(table: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(table) <= maximum:
        return table
    pieces = []
    for _, group in table.groupby("label"):
        fraction = len(group) / len(table)
        n = max(1, int(round(maximum * fraction)))
        pieces.append(group.sample(min(n, len(group)), random_state=seed))
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)
