"""Unit tests for PRG-AUC, event merging, and operating points."""

from __future__ import annotations

import numpy as np

from seizure_detector.metrics import (
    assign_event_ids,
    binary_metrics,
    bootstrap_metrics,
    event_metrics_from_windows,
    merge_windows_to_events,
    precision_at_recall,
    prevalence,
    prg_auc,
)


def test_prg_auc_always_positive_is_near_zero():
    y = np.array([0, 0, 0, 0, 1, 1])
    # Constant score → always-positive ranking → PRG area ≈ 0
    p = np.ones_like(y, dtype=float) * 0.5
    assert prg_auc(y, p) < 0.05


def test_prg_auc_perfect_is_high():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.95])
    assert prg_auc(y, p) > 0.99
    assert binary_metrics(y, p, 0.5)["pr_lift"] > 2.0


def test_precision_at_recall():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    point = precision_at_recall(y, p, 0.9)
    assert point["recall"] >= 0.9
    assert point["precision"] > 0.5


def test_event_merge_and_match():
    records = np.array(["a", "a", "a", "a"])
    starts = np.array([0.0, 5.0, 100.0, 105.0])
    durs = np.array([10.0, 10.0, 10.0, 10.0])
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.8, 0.1, 0.95])
    events = merge_windows_to_events(records, starts, durs, labels.astype(bool), merge_gap_seconds=30)
    assert len(events) == 1
    met = event_metrics_from_windows(
        records, starts, durs, labels, probs, threshold=0.5, duration_hours=1.0
    )
    assert met["n_ref_events"] == 1
    assert met["event_fp"] >= 1  # isolated high-prob negative becomes an event


def test_bootstrap_runs():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.7, 0.25])
    records = np.array(["r1", "r1", "r2", "r2", "r1", "r1", "r2", "r2"])
    eids = assign_event_ids(records, np.array([0, 5, 0, 5, 20, 25, 40, 45], dtype=float), y)
    boot = bootstrap_metrics(y, p, event_ids=eids, record_ids=records, n_boot=20, seed=1)
    assert "pr_auc" in boot
    assert boot["pr_auc"]["n"] > 0
    assert prevalence(y) == 0.375
