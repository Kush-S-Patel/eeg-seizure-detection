"""Official Precision-Recall-Gain curve (Flach & Kull).

Adapted from https://github.com/meeliskull/prg (MIT) for modern NumPy/Python.
Crossing-point construction is required for a correct AUPRG.
"""

from __future__ import annotations

import warnings

import numpy as np


def _precision(tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fp)


def _recall(tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fn)


def precision_gain(tp, fn, fp, tn):
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - (n_pos / n_neg) * (fp / tp)
    prec_gain = np.asarray(prec_gain, dtype=float)
    if prec_gain.ndim:
        prec_gain = prec_gain.copy()
        prec_gain[tn + fn == 0] = 0
    elif tn + fn == 0:
        prec_gain = 0.0
    return prec_gain


def recall_gain(tp, fn, fp, tn):
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        rg = 1.0 - (n_pos / n_neg) * (fn / tp)
    rg = np.asarray(rg, dtype=float)
    if rg.ndim:
        rg = rg.copy()
        rg[tn + fn == 0] = 1
    elif tn + fn == 0:
        rg = 1.0
    return rg


def _create_segments(labels, pos_scores, neg_scores):
    n = len(labels)
    new_order = np.lexsort((neg_scores, -pos_scores))
    labels = labels[new_order]
    pos_scores = pos_scores[new_order]
    neg_scores = neg_scores[new_order]
    segments = {
        "pos_score": np.zeros(n),
        "neg_score": np.zeros(n),
        "pos_count": np.zeros(n),
        "neg_count": np.zeros(n),
    }
    j = -1
    for i, label in enumerate(labels):
        if (
            i == 0
            or pos_scores[i - 1] != pos_scores[i]
            or neg_scores[i - 1] != neg_scores[i]
        ):
            j += 1
            segments["pos_score"][j] = pos_scores[i]
            segments["neg_score"][j] = neg_scores[i]
        if label == 0:
            segments["neg_count"][j] += 1
        else:
            segments["pos_count"][j] += 1
    for key in segments:
        segments[key] = segments[key][: j + 1]
    return segments


def _get_point(points, index):
    keys = list(points.keys())
    point = np.zeros(len(keys))
    key_indices = {}
    for i, key in enumerate(keys):
        point[i] = points[key][index]
        key_indices[key] = i
    return point, key_indices


def _insert_point(new_point, key_indices, points, precision_gain_v=0, recall_gain_v=0, is_crossing=0):
    for key in key_indices:
        points[key] = np.insert(points[key], 0, new_point[key_indices[key]])
    points["precision_gain"][0] = precision_gain_v
    points["recall_gain"][0] = recall_gain_v
    points["is_crossing"][0] = is_crossing
    new_order = np.lexsort((-points["precision_gain"], points["recall_gain"]))
    for key in points:
        points[key] = points[key][new_order]
    return points


def _create_crossing_points(points, n_pos, n_neg):
    n = n_pos + n_neg
    points["is_crossing"] = np.zeros(len(points["pos_score"]))
    j = int(np.amin(np.where(points["recall_gain"] >= 0)[0]))
    if points["recall_gain"][j] > 0:
        point_1, key_indices_1 = _get_point(points, j)
        point_2, _ = _get_point(points, j - 1)
        delta = point_1 - point_2
        if delta[key_indices_1["TP"]] > 0:
            alpha = (n_pos * n_pos / n - points["TP"][j - 1]) / delta[key_indices_1["TP"]]
        else:
            alpha = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha * delta
        new_prec_gain = precision_gain(
            new_point[key_indices_1["TP"]],
            new_point[key_indices_1["FN"]],
            new_point[key_indices_1["FP"]],
            new_point[key_indices_1["TN"]],
        )
        points = _insert_point(
            new_point, key_indices_1, points, precision_gain_v=new_prec_gain, is_crossing=1
        )

    x = points["recall_gain"]
    y = points["precision_gain"]
    temp_y_0 = np.append(y, 0)
    temp_0_y = np.append(0, y)
    temp_1_x = np.append(1, x)
    with np.errstate(invalid="ignore"):
        indices = np.where(np.logical_and((temp_y_0 * temp_0_y < 0), (temp_1_x >= 0)))[0]
    for i in indices:
        cross_x = x[i - 1] + (-y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        point_1, key_indices_1 = _get_point(points, i)
        point_2, _ = _get_point(points, i - 1)
        delta = point_1 - point_2
        if delta[key_indices_1["TP"]] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - points["TP"][i - 1]) / delta[
                key_indices_1["TP"]
            ]
        else:
            alpha = (n_neg / n_pos * points["TP"][i - 1] - points["FP"][i - 1]) / delta[
                key_indices_1["FP"]
            ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha * delta
        new_rec_gain = recall_gain(
            new_point[key_indices_1["TP"]],
            new_point[key_indices_1["FN"]],
            new_point[key_indices_1["FP"]],
            new_point[key_indices_1["TN"]],
        )
        points = _insert_point(
            new_point, key_indices_1, points, recall_gain_v=new_rec_gain, is_crossing=1
        )
        indices = indices + 1
        x = points["recall_gain"]
        y = points["precision_gain"]
        temp_y_0 = np.append(y, 0)
        temp_0_y = np.append(0, y)
        temp_1_x = np.append(1, x)
    return points


def create_prg_curve(labels, pos_scores, neg_scores=None):
    if neg_scores is None or len(neg_scores) == 0:
        neg_scores = -np.asarray(pos_scores, dtype=float)
    labels = np.asarray(labels)
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    n = len(labels)
    n_pos = float(np.sum(labels == 1))
    n_neg = float(n - n_pos)
    labels = 1 * (labels == 1)
    segments = _create_segments(labels, pos_scores, neg_scores)
    points: dict = {}
    points["pos_score"] = np.insert(segments["pos_score"], 0, np.inf)
    points["neg_score"] = np.insert(segments["neg_score"], 0, -np.inf)
    points["TP"] = np.insert(np.cumsum(segments["pos_count"]), 0, 0)
    points["FP"] = np.insert(np.cumsum(segments["neg_count"]), 0, 0)
    points["FN"] = n_pos - points["TP"]
    points["TN"] = n_neg - points["FP"]
    points["precision"] = _precision(points["TP"], points["FN"], points["FP"], points["TN"])
    points["recall"] = _recall(points["TP"], points["FN"], points["FP"], points["TN"])
    points["precision_gain"] = precision_gain(points["TP"], points["FN"], points["FP"], points["TN"])
    points["recall_gain"] = recall_gain(points["TP"], points["FN"], points["FP"], points["TN"])
    points = _create_crossing_points(points, n_pos, n_neg)
    with np.errstate(invalid="ignore"):
        points["in_unit_square"] = np.logical_and(
            points["recall_gain"] >= 0, points["precision_gain"] >= 0
        )
    return points


def calc_auprg(prg_curve) -> float:
    area = 0.0
    recall_gain_v = prg_curve["recall_gain"]
    precision_gain_v = prg_curve["precision_gain"]
    for i in range(1, len(recall_gain_v)):
        if (not np.isnan(recall_gain_v[i - 1])) and (recall_gain_v[i - 1] >= 0):
            width = recall_gain_v[i] - recall_gain_v[i - 1]
            height = (precision_gain_v[i] + precision_gain_v[i - 1]) / 2
            area += width * height
    return float(area)
