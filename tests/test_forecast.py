"""Tests for preictal forecasting labels and multi-domain features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from seizure_detector.features import MultiDomainHead, bandpower_features, phase_locking_value
from seizure_detector.forecast import (
    ForecastConfig,
    build_forecast_windows,
    estimate_onsets_from_detection_windows,
    label_forecast_window,
)
from seizure_detector.models import create_model


def test_onset_clustering():
    starts = np.array([0.0, 5.0, 10.0, 1000.0, 1005.0])
    labels = np.array([1, 1, 1, 1, 1])
    onsets = estimate_onsets_from_detection_windows(starts, labels, cluster_gap_seconds=60)
    assert len(onsets) == 2
    assert 0 < onsets[0] < 30
    assert 1000 < onsets[1] < 1030


def test_preictal_labeling_sop_sph():
    cfg = ForecastConfig(sop_seconds=1800, sph_seconds=300)
    onsets = np.array([3600.0])
    # 20 min before onset → inside SOP after SPH → preictal
    assert label_forecast_window(3600 - 1200, onsets, cfg) == 1
    # 2 min before → peri-ictal exclude
    assert label_forecast_window(3600 - 120, onsets, cfg) is None
    # 3 hours before → interictal
    assert label_forecast_window(3600 - 10800, onsets, cfg) == 0


def test_filter_onsets_first_and_gap():
    from seizure_detector.forecast import filter_onsets

    onsets = np.array([100.0, 200.0, 10_000.0])
    cfg = ForecastConfig(first_onset_only=True, min_inter_onset_gap_seconds=3600)
    assert filter_onsets(onsets, cfg).tolist() == [100.0]
    cfg2 = ForecastConfig(first_onset_only=False, min_inter_onset_gap_seconds=3600)
    assert filter_onsets(onsets, cfg2).tolist() == [100.0, 10_000.0]


def test_build_forecast_windows_from_toy():
    rows = []
    # One recording with a seizure cluster around t=3600s and interictal far away
    for t in [100.0, 200.0, 3580.0, 3590.0, 2000.0]:
        rows.append(
            {
                "record_id": "r1",
                "patient": 1,
                "session": 1,
                "split": "train",
                "edf_path": "x.edf",
                "start_seconds": t,
                "duration_seconds": 10.0,
                "label": 1 if t >= 3580 else 0,
                "label_source": "xltek_point_marker",
                "n_recording_seizure_markers": 1,
                "window_id": len(rows),
            }
        )
    # Far interictal negative
    rows.append(
        {
            "record_id": "r1",
            "patient": 1,
            "session": 1,
            "split": "train",
            "edf_path": "x.edf",
            "start_seconds": 100.0,
            "duration_seconds": 10.0,
            "label": 0,
            "label_source": "xltek_point_marker",
            "n_recording_seizure_markers": 1,
            "window_id": 99,
        }
    )
    # Add explicit preictal candidate at onset-20min = 2400 if onset~3595
    rows.append(
        {
            "record_id": "r1",
            "patient": 1,
            "session": 1,
            "split": "train",
            "edf_path": "x.edf",
            "start_seconds": 2400.0,
            "duration_seconds": 10.0,
            "label": 0,
            "label_source": "xltek_point_marker",
            "n_recording_seizure_markers": 1,
            "window_id": 100,
        }
    )
    detection = pd.DataFrame(rows)
    out = build_forecast_windows(
        detection,
        cfg=ForecastConfig(max_negatives_per_recording=50),
        output_path=__import__("pathlib").Path("_tmp_forecast_test.parquet"),
    )
    assert set(out["label"].unique()) <= {0, 1}
    assert "window_id" in out.columns
    __import__("pathlib").Path("_tmp_forecast_test.parquet").unlink(missing_ok=True)
    __import__("pathlib").Path("_tmp_forecast_test.json").unlink(missing_ok=True)


def test_multidomain_features_and_model():
    x = torch.randn(2, 18, 1280)
    bp = bandpower_features(x)
    assert bp.shape == (2, 18 * 5)
    plv = phase_locking_value(x)
    assert plv.shape == (2, 5)
    head = MultiDomainHead(channels=18, dim=32)
    emb = head(x)
    assert emb.shape[0] == 2
    model = create_model("eeg_conformer_multidomain", channels=18, samples=1280, dropout=0.1)
    model.eval()
    logits = model(x, torch.ones(2, 18))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
