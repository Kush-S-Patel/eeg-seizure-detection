"""Equal-weight ensemble of ft + ft2 + s42 with max-15 smooth + bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seizure_detector.engine import smooth_probabilities_by_recording
from seizure_detector.metrics import (
    assign_event_ids,
    binary_metrics,
    bootstrap_metrics,
    event_metrics_from_windows,
)

OUT = Path("outputs/rigorous_eval")
OUT.mkdir(parents=True, exist_ok=True)
PATHS = {
    "ft": Path("outputs/conformer_ft/test_predictions.csv"),
    "ft2": Path("outputs/conformer_ft2/test_predictions.csv"),
    "s42": Path("outputs/conformer_ft_s42/test_predictions.csv"),
}
VAL_PATHS = {
    "ft": Path("outputs/conformer_ft/val_predictions.csv"),
    "ft2": Path("outputs/conformer_ft2/val_predictions.csv"),
    "s42": Path("outputs/conformer_ft_s42/val_predictions.csv"),
}
KEY = ["record_id", "start_seconds"]


def blend(paths: dict[str, Path]) -> pd.DataFrame:
    dfs = {n: pd.read_csv(p) for n, p in paths.items() if p.exists()}
    names = list(dfs)
    base = dfs[names[0]][KEY + ["label", "duration_seconds"]].copy()
    base["probability"] = 0.0
    w = 1.0 / len(names)
    for n, d in dfs.items():
        m = d[KEY + ["probability"]].rename(columns={"probability": n})
        base = base.merge(m, on=KEY)
        base["probability"] += w * base[n]
    base["threshold"] = 0.5
    base["prediction"] = 0
    return smooth_probabilities_by_recording(base, smooth_seconds=15.0, mode="max", rethreshold=True)


def score(df: pd.DataFrame) -> dict:
    hours = float(df["duration_seconds"].sum() / 3600)
    y = df["label"].to_numpy()
    p = df["probability"].to_numpy()
    thr = float(df["threshold"].iloc[0])
    met = binary_metrics(y, p, thr, hours)
    met.update(
        event_metrics_from_windows(
            df["record_id"].to_numpy(),
            df["start_seconds"].to_numpy(),
            df["duration_seconds"].to_numpy(),
            y,
            p,
            thr,
            hours,
        )
    )
    eids = assign_event_ids(df["record_id"].to_numpy(), df["start_seconds"].to_numpy(), y)
    boot = bootstrap_metrics(y, p, event_ids=eids, record_ids=df["record_id"].to_numpy(), n_boot=100)
    met["boot_pr_auc_lo"] = boot["pr_auc"]["lo"]
    met["boot_pr_auc_hi"] = boot["pr_auc"]["hi"]
    met["boot_roc_auc_lo"] = boot["roc_auc"]["lo"]
    met["boot_roc_auc_hi"] = boot["roc_auc"]["hi"]
    return met


def main() -> None:
    for split, paths in [("test", PATHS), ("val", VAL_PATHS)]:
        if not all(p.exists() for p in paths.values()):
            print("skip", split, "missing csv")
            continue
        df = blend(paths)
        met = score(df)
        df.to_csv(OUT / f"{split}_ensemble_equal_max15.csv", index=False)
        (OUT / f"{split}_ensemble_equal_max15.json").write_text(
            json.dumps(met, indent=2, default=float), encoding="utf-8"
        )
        print(
            f"equal-ens {split}: PR={met['pr_auc']:.4f} [{met['boot_pr_auc_lo']:.3f},{met['boot_pr_auc_hi']:.3f}] "
            f"PRG={met['prg_auc']:.4f} ROC={met['roc_auc']:.4f} "
            f"P@R50={met['precision_at_recall_r50']:.3f} P@R70={met['precision_at_recall_r70']:.3f} "
            f"P@R90={met['precision_at_recall_r90']:.3f} "
            f"FA24h@R50={met['fp_per_24h_at_recall_r50']:.0f} "
            f"event_sens={met['event_sensitivity']:.3f} event_FA24h={met['event_fa_per_24h']:.1f}"
        )


if __name__ == "__main__":
    main()
