import numpy as np
import pandas as pd

from seizure_detector.dashboard import band_powers, evaluation_curves, stacked_eeg_figure
from seizure_detector.metrics import binary_metrics, choose_threshold


def test_metrics_and_threshold():
    targets = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold = choose_threshold(targets, probabilities)
    metrics = binary_metrics(targets, probabilities, threshold, duration_hours=1)
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_alarms_per_hour"] == 0


def test_dashboard_helpers():
    data = np.random.default_rng(2).normal(size=(18, 1280))
    assert len(stacked_eeg_figure(data, 128).data) == 18
    powers = band_powers(data[0], 128)
    assert set(powers["band"]) == {"delta", "theta", "alpha", "beta", "gamma"}
    predictions = pd.DataFrame({"label": [0, 1], "probability": [0.1, 0.9]})
    assert len(evaluation_curves(predictions).data) == 2
