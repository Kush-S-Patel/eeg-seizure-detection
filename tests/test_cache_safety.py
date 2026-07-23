"""Regression: subset train must not wipe a larger window cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seizure_detector.cache import allocate_cache, ensure_window_cache
from seizure_detector.config import SignalConfig


def _toy_windows(n: int, start_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": np.arange(start_id, start_id + n),
            "record_id": [f"r{i // 10}" for i in range(n)],
            "start_seconds": np.arange(n, dtype=float) * 5.0,
            "duration_seconds": np.full(n, 10.0),
            "edf_path": ["missing.edf"] * n,
            "label": np.zeros(n, dtype=int),
            "patient": np.zeros(n, dtype=int),
            "split": ["train"] * n,
        }
    )


def test_allocate_refuses_shrink(tmp_path: Path):
    cfg = SignalConfig(sample_rate=128)
    full = _toy_windows(100)
    allocate_cache(full, cfg, tmp_path)
    x = tmp_path / "x.f32"
    assert x.exists()
    # Poison a few bytes so we can detect wipe
    mm = np.memmap(x, dtype=np.float32, mode="r+", shape=(100, 18, 1280))
    mm[50] = 1.0
    mm.flush()
    del mm

    subset = _toy_windows(40)  # max window_id 39 → would shrink
    with pytest.raises(RuntimeError, match="shrink|wipe"):
        allocate_cache(subset, cfg, tmp_path, force=False)

    mm = np.memmap(x, dtype=np.float32, mode="r", shape=(100, 18, 1280))
    assert float(mm[50].mean()) == 1.0


def test_ensure_refuses_subset_rebuild_when_superset_exists(tmp_path: Path):
    cfg = SignalConfig(sample_rate=128)
    full = _toy_windows(100)
    allocate_cache(full, cfg, tmp_path)
    # Mark all records complete so covers would pass if data were real —
    # but leave progress empty for subset records check path.
    (tmp_path / "completed_records.json").write_text(
        json.dumps({"completed": []}), encoding="utf-8"
    )
    subset = full.iloc[::2].copy()
    with pytest.raises(RuntimeError, match="Refusing to reallocate"):
        ensure_window_cache(subset, cfg, tmp_path, rebuild=False)
