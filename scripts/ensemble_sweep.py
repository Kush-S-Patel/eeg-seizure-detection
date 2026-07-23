"""Grid-search ensemble weights on val, report best blend on test."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from seizure_detector.metrics import binary_metrics, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ensemble_max"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ft": ROOT / "outputs/conformer_ft/val_predictions.csv",
    "focal": ROOT / "outputs/conformer_focal/val_predictions.csv",
    "baseline": ROOT / "outputs/baseline/val_predictions.csv",
}
TEST_MODELS = {
    "ft": ROOT / "outputs/conformer_ft/test_predictions.csv",
    "focal": ROOT / "outputs/conformer_focal/test_predictions.csv",
    "baseline": ROOT / "outputs/baseline/test_predictions.csv",
}
KEY = ["record_id", "start_seconds"]


def _load(paths: dict[str, Path], split: str) -> pd.DataFrame:
    frames = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} predictions for {name}: {path}")
        frames[name] = pd.read_csv(path)
    base = frames[list(frames)[0]][KEY + ["label", "duration_seconds"]].copy()
    for name, df in frames.items():
        base = base.merge(
            df[KEY + ["probability"]].rename(columns={"probability": name}),
            on=KEY,
            how="inner",
        )
    return base


def _metrics(df: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    p = sum(weights[k] * df[k] for k in weights)
    hours = float(df["duration_seconds"].sum() / 3600)
    thr = choose_threshold(df["label"].to_numpy(), p.to_numpy())
    return binary_metrics(df["label"].to_numpy(), p.to_numpy(), thr, hours)


def main() -> None:
    val = _load(MODELS, "val")
    test = _load(TEST_MODELS, "test")
    names = list(MODELS)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    best_val = None
    best_weights = None
    for w_ft in grid:
        for w_focal in grid:
            w_base = 1.0 - w_ft - w_focal
            if w_base < -1e-9:
                continue
            weights = {"ft": w_ft, "focal": w_focal, "baseline": max(w_base, 0.0)}
            met = _metrics(val, weights)
            row = {**weights, **{f"val_{k}": v for k, v in met.items()}}
            rows.append(row)
            if best_val is None or met["pr_auc"] > best_val["pr_auc"]:
                best_val = met
                best_weights = weights

    assert best_weights is not None and best_val is not None
    test_met = _metrics(test, best_weights)
    summary = {
        "best_weights": best_weights,
        "val": best_val,
        "test": test_met,
    }
    (OUT / "ensemble_3way_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).head(20).to_csv(
        OUT / "ensemble_3way_top20_val.csv", index=False
    )
    print("BEST WEIGHTS:", best_weights)
    print(
        f"VAL  PR={best_val['pr_auc']:.4f} ROC={best_val['roc_auc']:.4f} "
        f"F1={best_val['f1']:.4f} sens={best_val['sensitivity']:.3f}"
    )
    print(
        f"TEST PR={test_met['pr_auc']:.4f} ROC={test_met['roc_auc']:.4f} "
        f"F1={test_met['f1']:.4f} sens={test_met['sensitivity']:.3f}"
    )


if __name__ == "__main__":
    main()
