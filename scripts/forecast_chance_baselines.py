"""Chance / permutation baselines for seizure forecasting (Mormann-style rigor).

Compares real forecast labels against:
  1) prevalence baseline (always predict prior)
  2) label-shuffle within split
  3) circular onset-shift re-labeling (structure-preserving null)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from seizure_detector.config import ARTIFACTS_DIR, WINDOWS_PATH
from seizure_detector.forecast import (
    FORECAST_WINDOWS_PATH,
    ForecastConfig,
    build_forecast_windows,
    chance_label_shuffle,
    estimate_onsets_from_detection_windows,
    label_forecast_window,
)
from seizure_detector.metrics import binary_metrics, prevalence

OUT = ARTIFACTS_DIR / "forecast_chance_baselines.json"


def _score_constant(labels: np.ndarray, prior: float) -> dict:
    probs = np.full(len(labels), prior, dtype=float)
    return binary_metrics(labels, probs, threshold=0.5)


def _onset_shift_null(detection: pd.DataFrame, cfg: ForecastConfig, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for record_id, group in detection.groupby("record_id", sort=False):
        starts = group["start_seconds"].to_numpy(dtype=float)
        dur = float(group["duration_seconds"].iloc[0])
        onsets = estimate_onsets_from_detection_windows(
            starts, group["label"].to_numpy(), window_seconds=dur
        )
        if len(onsets):
            rec_dur = float(starts.max() + dur + 1.0)
            shift = float(rng.uniform(0, max(rec_dur, 1.0)))
            onsets = np.sort((onsets + shift) % rec_dur)
        centers = starts + dur / 2.0
        for i, center in enumerate(centers):
            lab = label_forecast_window(float(center), onsets, cfg)
            if lab is None:
                continue
            row = group.iloc[i].to_dict()
            row["label"] = int(lab)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    forecast_path = FORECAST_WINDOWS_PATH
    if not forecast_path.exists():
        raise SystemExit(f"Missing {forecast_path}; run: seizure-detector prepare-forecast")
    forecast = pd.read_parquet(forecast_path)
    detection = pd.read_parquet(WINDOWS_PATH)
    cfg = ForecastConfig()
    report: dict = {"n_windows": int(len(forecast)), "by_split": {}}

    for split in ["train", "val", "test"]:
        y = forecast.loc[forecast["split"] == split, "label"].to_numpy(dtype=int)
        if not len(y) or len(np.unique(y)) < 2:
            continue
        prior = float(y.mean())
        const = _score_constant(y, prior)
        shuffled = chance_label_shuffle(pd.Series(y), seed=1337 + hash(split) % 1000)
        # Score shuffled labels against original probs=prior still → same; instead
        # report prevalence and that AP for random ranking equals prior.
        rng = np.random.default_rng(42)
        random_scores = rng.random(len(y))
        rand_met = binary_metrics(y, random_scores, 0.5)
        report["by_split"][split] = {
            "prevalence": prior,
            "constant_prior_pr_auc": const["pr_auc"],
            "random_score_pr_auc": rand_met["pr_auc"],
            "random_score_prg_auc": rand_met["prg_auc"],
            "n": int(len(y)),
            "n_pos": int(y.sum()),
        }

    # Onset-shift null on detection table (expensive-ish but offline)
    null = _onset_shift_null(detection, cfg, seed=2024)
    if len(null) and null["label"].nunique() > 1:
        y = null["label"].to_numpy(dtype=int)
        rng = np.random.default_rng(7)
        report["onset_shift_null"] = {
            "n": int(len(null)),
            "prevalence": float(y.mean()),
            "random_score_pr_auc": binary_metrics(y, rng.random(len(y)), 0.5)["pr_auc"],
            "by_split": null.groupby("split")["label"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "n", "sum": "n_pos", "mean": "prevalence"})
            .to_dict(orient="index"),
        }

    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
