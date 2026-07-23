from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seizure_detector.config import WindowConfig
from seizure_detector.paths import load_splits
from seizure_detector import windows as windows_module


def test_patient_leakage_rejected(tmp_path):
    path = tmp_path / "splits.parquet"
    pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "patient": [1, 1],
            "session": [1, 2],
            "split": ["train", "test"],
        }
    ).to_csv(path.with_suffix(".csv"), index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_splits(path)


def test_ambiguous_windows_are_excluded(tmp_path, monkeypatch):
    annotation = tmp_path / "events.csv"
    annotation.write_text("Text,CreationTime\n@Seizure,2023-01-01T00:08:20\n", encoding="utf-8")
    paths = {
        "edf": tmp_path / "signal.edf",
        "xltek": annotation,
        "json": tmp_path / "x.json",
        "channels": tmp_path / "x.tsv",
    }
    monkeypatch.setattr(
        windows_module,
        "recording_info",
        lambda *_, **__: {"duration_seconds": 1000.0, "start": "2023-01-01T00:00:00"},
    )
    monkeypatch.setattr(
        windows_module,
        "read_seizure_times",
        lambda *_: np.asarray([500.0]),
    )
    cfg = WindowConfig(
        window_seconds=10,
        stride_seconds=10,
        positive_radius_seconds=20,
        negative_guard_seconds=100,
        max_negative_windows_per_recording=1000,
    )
    rec = SimpleNamespace(record_id="r", patient=1, session=1, split="train")
    rows = windows_module._window_rows(rec, paths, cfg, np.random.default_rng(1))
    centers = np.asarray([r["start_seconds"] + 5 for r in rows])
    labels = np.asarray([r["label"] for r in rows])
    assert np.all(np.abs(centers[labels == 1] - 500) <= 20)
    assert np.all(np.abs(centers[labels == 0] - 500) >= 100)
