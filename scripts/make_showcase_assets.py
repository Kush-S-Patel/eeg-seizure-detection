"""Build lightweight showcase assets for README + notebooks (no 244GB cache needed)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SHOW = ROOT / "docs" / "showcase"
METRICS = SHOW / "metrics"
FIGS = SHOW / "figures"
PRED = ROOT / "outputs" / "conformer_ft" / "test_predictions_smooth15s.csv"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 140,
        }
    )


def export_metrics() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    mapping = {
        "headline.json": ROOT / "outputs" / "rigorous_eval" / "headline.json",
        "split_audit.json": ROOT / "outputs" / "rigorous_eval" / "split_audit.json",
        "conformer_ft_test_smooth15.json": ROOT
        / "outputs"
        / "conformer_ft"
        / "test_metrics_smooth15s.json",
        "conformer_ft_test_raw.json": ROOT / "outputs" / "conformer_ft" / "test_metrics.json",
        "conformer_ft_val_smooth15.json": ROOT
        / "outputs"
        / "conformer_ft"
        / "val_metrics_smooth15s.json",
        "baseline_test.json": ROOT / "outputs" / "baseline" / "test_metrics.json",
        "conformer_ft_history.csv": ROOT / "outputs" / "conformer_ft" / "history.csv",
        "forecast_v2_summary.json": ROOT / "outputs" / "forecast_v2" / "summary.json",
        "spectral_lr_test.json": ROOT / "docs" / "showcase" / "metrics" / "spectral_lr_test.json",
    }
    for name, src in mapping.items():
        if not src.exists():
            print(f"  skip missing {src}")
            continue
        dst = METRICS / name
        dst.write_bytes(src.read_bytes())
        print(f"  wrote {dst.relative_to(ROOT)}")


def export_scores() -> Path:
    """Compact y_true / y_score for curve plotting (~few MB)."""
    METRICS.mkdir(parents=True, exist_ok=True)
    out = METRICS / "test_scores_smooth15.npz"
    if not PRED.exists():
        raise FileNotFoundError(PRED)
    df = pd.read_csv(PRED, usecols=["label", "probability", "patient", "start_seconds", "record_id"])
    y = df["label"].to_numpy(dtype=np.int8)
    p = df["probability"].to_numpy(dtype=np.float32)
    # Stratified subsample for notebooks that want faster plots (full curves use all).
    rng = np.random.default_rng(1337)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(len(pos), 20_000)
    n_neg = min(len(neg), 80_000)
    idx = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(idx)
    np.savez_compressed(
        out,
        y_full=y,
        p_full=p,
        y_sample=y[idx],
        p_sample=p[idx],
        patient_sample=df["patient"].to_numpy()[idx],
        start_sample=df["start_seconds"].to_numpy(dtype=np.float32)[idx],
        record_sample=df["record_id"].astype("U")[idx],
        n_windows=np.int64(len(df)),
        prevalence=np.float64(y.mean()),
        pr_auc=np.float64(average_precision_score(y, p)),
        roc_auc=np.float64(roc_auc_score(y, p)),
    )
    print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def make_figures(scores_path: Path) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    _style()
    data = np.load(scores_path, allow_pickle=False)
    y = data["y_full"]
    p = data["p_full"]
    prev = float(data["prevalence"])

    # 1) PR curve
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    PrecisionRecallDisplay.from_predictions(y, p, ax=ax, name="Conformer (smooth 15s)")
    ax.axhline(prev, color="#888888", ls="--", lw=1.2, label=f"Chance (prevalence={prev:.3f})")
    ax.set_title("Test precision–recall (seizure detection)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "pr_curve_test.png", bbox_inches="tight")
    plt.close(fig)

    # 2) ROC curve
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    RocCurveDisplay.from_predictions(y, p, ax=ax, name="Conformer (smooth 15s)")
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1.2, label="Chance")
    ax.set_title("Test ROC (seizure detection)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "roc_curve_test.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Model comparison bar chart
    headline = json.loads((METRICS / "headline.json").read_text(encoding="utf-8"))
    baseline = json.loads((METRICS / "baseline_test.json").read_text(encoding="utf-8"))
    ft_raw = json.loads((METRICS / "conformer_ft_test_raw.json").read_text(encoding="utf-8"))
    ft_s = json.loads((METRICS / "conformer_ft_test_smooth15.json").read_text(encoding="utf-8"))
    lr_path = METRICS / "spectral_lr_test.json"
    names = ["Bandpower\nLogReg", "EEGNet\nCNN", "Conformer\nraw", "Conformer\n+15s smooth"]
    prs = [baseline["pr_auc"], ft_raw["pr_auc"], ft_s["pr_auc"]]
    rocs = [baseline["roc_auc"], ft_raw["roc_auc"], ft_s["roc_auc"]]
    if lr_path.exists():
        lr = json.loads(lr_path.read_text(encoding="utf-8"))
        prs = [lr["pr_auc"], baseline["pr_auc"], ft_raw["pr_auc"], ft_s["pr_auc"]]
        rocs = [lr["roc_auc"], baseline["roc_auc"], ft_raw["roc_auc"], ft_s["roc_auc"]]
    else:
        names = names[1:]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    b1 = ax.bar(x - width / 2, prs, width, label="PR-AUC", color="#1f6f8b")
    b2 = ax.bar(x + width / 2, rocs, width, label="ROC-AUC", color="#99c24d")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test performance: classical → Conformer")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(FIGS / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 4) Operating points from rigorous headline numbers
    ops = headline["test_ft_max15"]
    labels = ["R50", "R70", "R90"]
    prec = [
        ops["precision_at_recall_50"],
        ops["precision_at_recall_70"],
        ops["precision_at_recall_90"],
    ]
    # Headline stores FA at R50/R90; R70 FA taken from published summary block for ft max15.
    fa = [
        ops["fp_per_24h_at_recall_50"],
        1265.0,  # ft_test_max15 from rigorous_eval/summary.json
        ops["fp_per_24h_at_recall_90"],
    ]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.6))
    ax1.bar(labels, prec, color="#1f6f8b", label="Precision")
    ax1.set_ylabel("Precision")
    ax1.set_ylim(0, 0.45)
    ax2 = ax1.twinx()
    ax2.plot(labels, fa, color="#c73e1d", marker="o", lw=2, label="False alarms / 24h")
    ax2.set_ylabel("False alarms per 24h")
    ax1.set_title("Clinical operating points (test, smoothed)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "operating_points.png", bbox_inches="tight")
    plt.close(fig)

    # 5) Confusion matrix at F1 threshold from metrics json
    thr = float(ft_s["threshold"])
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black", fontsize=11)
    ax.set_title(f"Confusion matrix (thr={thr:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 6) Training curve
    hist = pd.read_csv(METRICS / "conformer_ft_history.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(hist["epoch"], hist["val_pr_auc"], marker="o", color="#1f6f8b", label="Val PR-AUC")
    ax.plot(hist["epoch"], hist["train_loss"], marker="s", color="#888888", label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Fine-tune training trajectory (conformer_ft)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "training_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 7) Pipeline schematic (matplotlib boxes)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (0.3, "BDSP / S3\nNeurotech EEG"),
        (2.3, "Patient splits\n+ weak labels"),
        (4.3, "Window cache\n18ch × 10s"),
        (6.3, "EEG Conformer\n+ focal loss"),
        (8.3, "PR / PRG /\nevent metrics"),
    ]
    for x0, text in steps:
        ax.add_patch(
            plt.Rectangle((x0, 0.7), 1.6, 1.5, fill=True, facecolor="#e8f1f5", edgecolor="#1f6f8b", lw=1.5)
        )
        ax.text(x0 + 0.8, 1.45, text, ha="center", va="center", fontsize=9)
    for x0 in (1.9, 3.9, 5.9, 7.9):
        ax.annotate("", xy=(x0 + 0.35, 1.45), xytext=(x0, 1.45), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("End-to-end pipeline", pad=8)
    fig.tight_layout()
    fig.savefig(FIGS / "pipeline.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote figures under {FIGS.relative_to(ROOT)}")


def main() -> None:
    print("Exporting showcase metrics...")
    export_metrics()
    print("Exporting compact scores...")
    scores = export_scores()
    print("Rendering figures...")
    make_figures(scores)
    print("Done.")


if __name__ == "__main__":
    main()
