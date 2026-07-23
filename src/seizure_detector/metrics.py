"""Precision-recall-gain, operating points, and event-level evaluation.

PRG-AUC follows Flach & Kull (ICML 2015): rescale precision/recall so the
always-positive baseline sits at 0, with linear interpolation in PRG space
(valid under the paper's construction; raw PR linear interpolation is not).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .prg import calc_auprg, create_prg_curve


def choose_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return 0.5
    probabilities = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def prevalence(targets: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=int)
    return float(targets.mean()) if len(targets) else float("nan")


def _precision_gain(precision: np.ndarray, pi: float) -> np.ndarray:
    denom = (1.0 - pi) * np.maximum(precision, 1e-12)
    return (precision - pi) / denom


def _recall_gain(recall: np.ndarray, pi: float) -> np.ndarray:
    denom = (1.0 - pi) * np.maximum(recall, 1e-12)
    return (recall - pi) / denom


def prg_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the Precision-Recall-Gain curve (Flach & Kull)."""
    targets = np.asarray(targets, dtype=int)
    probabilities = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    if len(np.unique(targets)) < 2:
        return float("nan")
    pi = float(targets.mean())
    if pi <= 0.0 or pi >= 1.0:
        return float("nan")
    curve = create_prg_curve(targets, probabilities)
    return float(calc_auprg(curve))


def precision_at_recall(
    targets: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
) -> dict[str, float]:
    """Highest precision achievable at recall ≥ target_recall."""
    targets = np.asarray(targets, dtype=int)
    probabilities = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    # precision/recall arrays end with (1.0 precision at recall=0) convention from sklearn:
    # thresholds align with precision[:-1], recall[:-1].
    ok = recall[:-1] >= target_recall
    if not ok.any():
        return {
            "target_recall": float(target_recall),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": float("nan"),
        }
    idx = int(np.nanargmax(np.where(ok, precision[:-1], -np.inf)))
    return {
        "target_recall": float(target_recall),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "threshold": float(thresholds[idx]),
    }


def operating_points(
    targets: np.ndarray,
    probabilities: np.ndarray,
    duration_hours: float,
    recalls: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
) -> dict[str, float]:
    """Precision / FP rates at stated recall targets (window-level)."""
    out: dict[str, float] = {}
    n = len(targets)
    for r in recalls:
        point = precision_at_recall(targets, probabilities, r)
        thr = point["threshold"]
        key = f"r{int(round(r * 100)):02d}"
        out[f"precision_at_recall_{key}"] = point["precision"]
        out[f"recall_at_recall_{key}"] = point["recall"]
        out[f"threshold_at_recall_{key}"] = thr
        if np.isfinite(thr):
            pred = (probabilities >= thr).astype(int)
            fp = int(((pred == 1) & (targets == 0)).sum())
            out[f"fp_per_24h_at_recall_{key}"] = float(fp / max(duration_hours, 1e-9) * 24.0)
            out[f"fp_per_hour_at_recall_{key}"] = float(fp / max(duration_hours, 1e-9))
        else:
            out[f"fp_per_24h_at_recall_{key}"] = float("nan")
            out[f"fp_per_hour_at_recall_{key}"] = float("nan")
    out["n_windows"] = float(n)
    out["prevalence"] = prevalence(targets)
    return out


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    duration_hours: float | None = None,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    two_classes = len(np.unique(targets)) == 2
    result = {
        "pr_auc": float(average_precision_score(targets, probabilities)) if two_classes else float("nan"),
        "prg_auc": float(prg_auc(targets, probabilities)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(targets, probabilities)) if two_classes else float("nan"),
        "prevalence": prevalence(targets),
        "pr_lift": (
            float(average_precision_score(targets, probabilities) / max(prevalence(targets), 1e-12))
            if two_classes
            else float("nan")
        ),
        "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "brier": float(brier_score_loss(targets, probabilities)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if duration_hours is not None:
        result["false_alarms_per_hour"] = float(fp / max(duration_hours, 1e-9))
        result["false_alarms_per_24h"] = float(fp / max(duration_hours, 1e-9) * 24.0)
        result.update(operating_points(targets, probabilities, duration_hours))
    return result


@dataclass(frozen=True)
class EventInterval:
    record_id: str
    start: float
    end: float
    event_id: int


def merge_windows_to_events(
    record_ids: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    positive_mask: np.ndarray,
    *,
    merge_gap_seconds: float = 30.0,
    min_duration_seconds: float = 0.0,
    pre_tolerance: float = 0.0,
    post_tolerance: float = 0.0,
) -> list[EventInterval]:
    """Merge positive windows into SzCORE-style events with gap + min duration."""
    events: list[EventInterval] = []
    eid = 0
    order = np.argsort(np.asarray(record_ids, dtype=object), kind="mergesort")
    # Stable secondary sort by start within records via lexsort.
    starts_a = np.asarray(starts, dtype=float)
    order = np.lexsort((starts_a, np.asarray(record_ids, dtype=object)))
    rec = np.asarray(record_ids, dtype=object)[order]
    st = starts_a[order]
    dur = np.asarray(durations, dtype=float)[order]
    pos = np.asarray(positive_mask, dtype=bool)[order]
    i = 0
    n = len(rec)
    while i < n:
        if not pos[i]:
            i += 1
            continue
        rid = rec[i]
        seg_start = st[i]
        seg_end = st[i] + dur[i]
        j = i + 1
        while j < n and rec[j] == rid and pos[j] and st[j] <= seg_end + merge_gap_seconds:
            seg_end = max(seg_end, st[j] + dur[j])
            j += 1
        if (seg_end - seg_start) >= min_duration_seconds:
            events.append(
                EventInterval(
                    record_id=str(rid),
                    start=float(seg_start - pre_tolerance),
                    end=float(seg_end + post_tolerance),
                    event_id=eid,
                )
            )
            eid += 1
        i = j
    return events


def events_overlap(a: EventInterval, b: EventInterval) -> bool:
    if a.record_id != b.record_id:
        return False
    return a.start < b.end and b.start < a.end


def match_events(
    reference: list[EventInterval],
    predicted: list[EventInterval],
) -> dict[str, float]:
    """Any-overlap matching (SzCORE-style simplified): greedy one-to-one."""
    ref_hit = np.zeros(len(reference), dtype=bool)
    pred_hit = np.zeros(len(predicted), dtype=bool)
    for i, p in enumerate(predicted):
        for j, r in enumerate(reference):
            if ref_hit[j]:
                continue
            if events_overlap(p, r):
                ref_hit[j] = True
                pred_hit[i] = True
                break
    tp = int(ref_hit.sum())
    fn = int((~ref_hit).sum())
    fp = int((~pred_hit).sum())
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "event_tp": float(tp),
        "event_fn": float(fn),
        "event_fp": float(fp),
        "event_sensitivity": float(sens),
        "event_precision": float(prec),
        "n_ref_events": float(len(reference)),
        "n_pred_events": float(len(predicted)),
    }


def event_metrics_from_windows(
    record_ids: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    duration_hours: float,
    *,
    merge_gap_seconds: float = 30.0,
    min_duration_seconds: float = 10.0,
    pre_tolerance: float = 30.0,
    post_tolerance: float = 30.0,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    pred = probs >= threshold
    ref = merge_windows_to_events(
        record_ids,
        starts,
        durations,
        labels.astype(bool),
        merge_gap_seconds=merge_gap_seconds,
        min_duration_seconds=0.0,
        pre_tolerance=0.0,
        post_tolerance=0.0,
    )
    hyp = merge_windows_to_events(
        record_ids,
        starts,
        durations,
        pred,
        merge_gap_seconds=merge_gap_seconds,
        min_duration_seconds=min_duration_seconds,
        pre_tolerance=pre_tolerance,
        post_tolerance=post_tolerance,
    )
    matched = match_events(ref, hyp)
    matched["event_fa_per_24h"] = float(matched["event_fp"] / max(duration_hours, 1e-9) * 24.0)
    matched["merge_gap_seconds"] = float(merge_gap_seconds)
    matched["min_duration_seconds"] = float(min_duration_seconds)
    matched["pre_tolerance"] = float(pre_tolerance)
    matched["post_tolerance"] = float(post_tolerance)
    return matched


def assign_event_ids(
    record_ids: np.ndarray,
    starts: np.ndarray,
    labels: np.ndarray,
    *,
    merge_gap_seconds: float = 60.0,
) -> np.ndarray:
    """Assign event ids to positive windows; negatives get -1."""
    n = len(labels)
    event_ids = np.full(n, -1, dtype=int)
    order = np.lexsort((np.asarray(starts, dtype=float), np.asarray(record_ids, dtype=object)))
    rec = np.asarray(record_ids, dtype=object)[order]
    st = np.asarray(starts, dtype=float)[order]
    lab = np.asarray(labels, dtype=int)[order]
    eid = 0
    i = 0
    while i < n:
        if lab[i] != 1:
            i += 1
            continue
        rid = rec[i]
        last = st[i]
        event_ids[order[i]] = eid
        j = i + 1
        while j < n and rec[j] == rid and lab[j] == 1 and st[j] - last <= merge_gap_seconds:
            event_ids[order[j]] = eid
            last = st[j]
            j += 1
        eid += 1
        i = j
    return event_ids


def bootstrap_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    event_ids: np.ndarray | None = None,
    record_ids: np.ndarray | None = None,
    duration_hours: float | None = None,
    n_boot: int = 200,
    seed: int = 1337,
    threshold: float | None = None,
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs by resampling seizure events (+ negative recordings).

    Positive windows are resampled as whole events. Negative windows are
    resampled by recording. Uses a compact per-unit representation so each
    bootstrap draw avoids repeatedly scanning the full window table.
    """
    del duration_hours, threshold  # ranking metrics only; thresholded ops omitted for speed
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets, dtype=int)
    probabilities = np.nan_to_num(np.asarray(probabilities, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    if event_ids is None:
        if record_ids is None:
            raise ValueError("Need event_ids or record_ids for bootstrap")
        event_ids = assign_event_ids(record_ids, np.arange(len(targets)), targets)
    event_ids = np.asarray(event_ids, dtype=int)
    if record_ids is None:
        record_ids = np.asarray(["_"] * len(targets), dtype=object)
    else:
        record_ids = np.asarray(record_ids, dtype=object)

    pos_events = np.unique(event_ids[event_ids >= 0])
    neg_mask = targets == 0
    neg_records = np.unique(record_ids[neg_mask])

    # Pre-bundle y/p for each resampling unit (single pass).
    event_y: dict[int, list[int]] = {int(e): [] for e in pos_events}
    event_p: dict[int, list[float]] = {int(e): [] for e in pos_events}
    record_y: dict[object, list[int]] = {r: [] for r in neg_records}
    record_p: dict[object, list[float]] = {r: [] for r in neg_records}
    for i in range(len(targets)):
        eid = int(event_ids[i])
        if eid >= 0:
            event_y[eid].append(int(targets[i]))
            event_p[eid].append(float(probabilities[i]))
        elif neg_mask[i]:
            rid = record_ids[i]
            record_y[rid].append(int(targets[i]))
            record_p[rid].append(float(probabilities[i]))
    event_bundles = {
        e: (np.asarray(event_y[e], dtype=int), np.asarray(event_p[e], dtype=float))
        for e in event_y
    }
    record_bundles = {
        r: (np.asarray(record_y[r], dtype=int), np.asarray(record_p[r], dtype=float))
        for r in record_y
    }

    samples: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "pr_lift": [],
    }
    for _ in range(n_boot):
        y_parts: list[np.ndarray] = []
        p_parts: list[np.ndarray] = []
        if len(pos_events):
            for e in rng.choice(pos_events, size=len(pos_events), replace=True):
                y_e, p_e = event_bundles[int(e)]
                y_parts.append(y_e)
                p_parts.append(p_e)
        if len(neg_records):
            for r in rng.choice(neg_records, size=len(neg_records), replace=True):
                y_r, p_r = record_bundles[r]
                y_parts.append(y_r)
                p_parts.append(p_r)
        if not y_parts:
            continue
        y = np.concatenate(y_parts)
        p = np.concatenate(p_parts)
        if len(np.unique(y)) < 2:
            continue
        pr = float(average_precision_score(y, p))
        samples["pr_auc"].append(pr)
        samples["roc_auc"].append(float(roc_auc_score(y, p)))
        samples["pr_lift"].append(pr / max(float(y.mean()), 1e-12))

    summary: dict[str, dict[str, float]] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        if not len(arr):
            summary[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0.0}
            continue
        summary[k] = {
            "mean": float(arr.mean()),
            "lo": float(np.quantile(arr, 0.025)),
            "hi": float(np.quantile(arr, 0.975)),
            "n": float(len(arr)),
        }
    summary["n_pos_events"] = {
        "mean": float(len(pos_events)),
        "lo": float(len(pos_events)),
        "hi": float(len(pos_events)),
        "n": 1.0,
    }
    summary["n_neg_records"] = {
        "mean": float(len(neg_records)),
        "lo": float(len(neg_records)),
        "hi": float(len(neg_records)),
        "n": 1.0,
    }
    return summary
