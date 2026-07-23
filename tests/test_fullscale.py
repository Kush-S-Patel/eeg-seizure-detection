"""Tests for full-scale pipeline helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from seizure_detector.cache import allocate_cache, load_progress
from seizure_detector.config import SignalConfig


def test_allocate_and_progress(tmp_path: Path):
    windows = pd.DataFrame(
        {
            "window_id": [0, 1],
            "record_id": ["rec-a", "rec-a"],
            "start_seconds": [0.0, 5.0],
            "duration_seconds": [10.0, 10.0],
            "edf_path": ["/tmp/a.edf", "/tmp/a.edf"],
            "label": [0, 1],
        }
    )
    cache_dir = tmp_path / "cache"
    allocate_cache(windows, SignalConfig(sample_rate=128), cache_dir)
    assert (cache_dir / "meta.json").exists()
    assert (cache_dir / "x.f32").exists()
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_rows"] == 2
    assert load_progress(cache_dir) == set()


def test_batch_chunking():
    from pipeline.fullscale import _batch_chunks

    gb = 1024**3
    selection = pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3"],
            "patient": [1, 1, 2],
            "session": [1, 2, 1],
            "size_bytes": [60 * gb, 60 * gb, 60 * gb],
        }
    )
    chunks = _batch_chunks(selection, batch_gb=150)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1
