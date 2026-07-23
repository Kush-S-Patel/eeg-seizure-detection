"""Apply max temporal pooling to saved prediction CSVs and score metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seizure_detector.engine import smooth_probabilities_by_recording
from seizure_detector.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[1]


def score_csv(path: Path, smooth_seconds: float = 15.0, mode: str = "max") -> dict:
    df = pd.read_csv(path)
    df["threshold"] = 0.5
    df["prediction"] = 0
    smoothed = smooth_probabilities_by_recording(
        df, smooth_seconds=smooth_seconds, mode=mode, rethreshold=True
    )
    hours = smoothed["duration_seconds"].sum() / 3600
    return binary_metrics(
        smoothed["label"].to_numpy(),
        smoothed["probability"].to_numpy(),
        float(smoothed["threshold"].iloc[0]),
        hours,
    )


def main() -> None:
    files = [
        ROOT / "outputs/conformer_ft/test_predictions.csv",
        ROOT / "outputs/conformer_focal/test_predictions.csv",
        ROOT / "outputs/baseline/test_predictions.csv",
    ]
    out = {}
    for path in files:
        if not path.exists():
            continue
        met = score_csv(path)
        out[path.parent.name] = met
        print(
            f"{path.parent.name}: PR={met['pr_auc']:.4f} ROC={met['roc_auc']:.4f} "
            f"F1={met['f1']:.4f}"
        )
    (ROOT / "outputs/ensemble_max/max_smooth_single_models.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
