"""Tests for overlapping-window probability smoothing."""

from __future__ import annotations

import pandas as pd

from seizure_detector.engine import smooth_probabilities_by_recording


def test_smooth_probabilities_averages_nearby_windows():
    predictions = pd.DataFrame(
        {
            "record_id": ["a", "a", "a", "b"],
            "start_seconds": [0.0, 5.0, 10.0, 0.0],
            "duration_seconds": [10.0, 10.0, 10.0, 10.0],
            "probability": [0.0, 1.0, 0.0, 0.9],
            "threshold": [0.5, 0.5, 0.5, 0.5],
            "label": [0, 1, 0, 1],
            "prediction": [0, 1, 0, 1],
        }
    )
    smoothed = smooth_probabilities_by_recording(
        predictions, smooth_seconds=15.0, mode="mean", rethreshold=False
    )
    # Middle window at t=5 should average all three from record a (radius 7.5).
    assert abs(smoothed.loc[1, "probability"] - (0.0 + 1.0 + 0.0) / 3) < 1e-6
    # Record b unchanged (only one window).
    assert abs(smoothed.loc[3, "probability"] - 0.9) < 1e-6
    assert "probability_raw" in smoothed.columns


def test_smooth_max_keeps_peak():
    predictions = pd.DataFrame(
        {
            "record_id": ["a", "a", "a"],
            "start_seconds": [0.0, 5.0, 10.0],
            "probability": [0.1, 0.9, 0.2],
            "threshold": [0.5, 0.5, 0.5],
            "label": [0, 1, 0],
            "prediction": [0, 1, 0],
        }
    )
    smoothed = smooth_probabilities_by_recording(
        predictions, smooth_seconds=15.0, mode="max", rethreshold=False
    )
    assert abs(smoothed.loc[0, "probability"] - 0.9) < 1e-6
    assert abs(smoothed.loc[1, "probability"] - 0.9) < 1e-6


def test_smooth_seconds_zero_is_noop():
    predictions = pd.DataFrame(
        {
            "record_id": ["a", "a"],
            "start_seconds": [0.0, 5.0],
            "probability": [0.1, 0.9],
            "threshold": [0.5, 0.5],
            "label": [0, 1],
            "prediction": [0, 1],
        }
    )
    out = smooth_probabilities_by_recording(predictions, smooth_seconds=0.0)
    assert list(out["probability"]) == [0.1, 0.9]
