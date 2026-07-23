"""Rigorous evaluation suite: PRG, bootstrap CIs, SzCORE events, ensemble, metadata.

Uses existing prediction CSVs when present to avoid re-running GPU inference;
falls back to checkpoint inference for missing models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from seizure_detector.config import ROOT
from seizure_detector.dataset import load_windows
from seizure_detector.engine import (
    evaluate_ensemble,
    smooth_probabilities_by_recording,
)
from seizure_detector.metrics import (
    assign_event_ids,
    binary_metrics,
    bootstrap_metrics,
    choose_threshold,
    event_metrics_from_windows,
    prevalence,
)

OUT = ROOT / "outputs" / "rigorous_eval"
OUT.mkdir(parents=True, exist_ok=True)

PRED_CANDIDATES = {
    "ft": ROOT / "outputs/conformer_ft/test_predictions.csv",
    "ft2": ROOT / "outputs/conformer_ft2/test_predictions.csv",
    "s42": ROOT / "outputs/conformer_ft_s42/test_predictions.csv",
    "focal": ROOT / "outputs/conformer_focal/test_predictions.csv",
    "baseline": ROOT / "outputs/baseline/test_predictions.csv",
}
VAL_CANDIDATES = {
    "ft": ROOT / "outputs/conformer_ft/val_predictions.csv",
    "ft2": ROOT / "outputs/conformer_ft2/val_predictions.csv",
    "s42": ROOT / "outputs/conformer_ft_s42/val_predictions.csv",
    "focal": ROOT / "outputs/conformer_focal/val_predictions.csv",
    "baseline": ROOT / "outputs/baseline/val_predictions.csv",
}
CKPTS = {
    "ft": ROOT / "outputs/conformer_ft/best.pt",
    "ft2": ROOT / "outputs/conformer_ft2/best.pt",
    "s42": ROOT / "outputs/conformer_ft_s42/best.pt",
}


def _load_pred(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _apply_max_smooth(df: pd.DataFrame, seconds: float = 15.0) -> pd.DataFrame:
    out = df.copy()
    if "threshold" not in out.columns:
        out["threshold"] = 0.5
    if "prediction" not in out.columns:
        out["prediction"] = 0
    return smooth_probabilities_by_recording(
        out, smooth_seconds=seconds, mode="max", rethreshold=True
    )


def _score(df: pd.DataFrame, *, n_boot: int = 100, tag: str = "") -> dict:
    hours = float(df["duration_seconds"].sum() / 3600)
    thr = float(df["threshold"].iloc[0]) if "threshold" in df.columns else choose_threshold(
        df["label"].to_numpy(), df["probability"].to_numpy()
    )
    y = df["label"].to_numpy()
    p = df["probability"].to_numpy()
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
    boot = bootstrap_metrics(
        y,
        p,
        event_ids=eids,
        record_ids=df["record_id"].to_numpy(),
        duration_hours=hours,
        n_boot=n_boot,
        threshold=thr,
    )
    for key, stats in boot.items():
        met[f"boot_{key}_mean"] = stats["mean"]
        met[f"boot_{key}_lo"] = stats["lo"]
        met[f"boot_{key}_hi"] = stats["hi"]
    met["n_pos_events"] = float(len(np.unique(eids[eids >= 0])))
    met["tag"] = tag
    return met


def _audit_splits(windows: pd.DataFrame) -> dict:
    report: dict = {"patient_overlap": {}, "prevalence": {}, "events": {}}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        pa = set(windows.loc[windows["split"] == a, "patient"])
        pb = set(windows.loc[windows["split"] == b, "patient"])
        report["patient_overlap"][f"{a}_vs_{b}"] = len(pa & pb)
    for split in ["train", "val", "test"]:
        s = windows[windows["split"] == split]
        report["prevalence"][split] = {
            "n_windows": int(len(s)),
            "n_pos": int(s["label"].sum()),
            "prevalence": float(s["label"].mean()),
            "n_patients": int(s["patient"].nunique()),
            "n_records": int(s["record_id"].nunique()),
            "marker_sum": int(s.groupby("record_id")["n_recording_seizure_markers"].max().sum()),
        }
        eids = assign_event_ids(
            s["record_id"].to_numpy(),
            s["start_seconds"].to_numpy(),
            s["label"].to_numpy(),
        )
        report["events"][split] = int(len(np.unique(eids[eids >= 0])))
    return report


def _load_phenotype() -> pd.DataFrame:
    pheno = ROOT / "data/raw/EEG/bids/Neurotech/phenotype"
    demo = pd.read_csv(pheno / "demographics.tsv", sep="\t")
    findings = pd.read_csv(pheno / "eeg_findings.tsv", sep="\t")
    # participant_id like sub-Neurotech1 → patient int
    def pid(s: str) -> int:
        return int(str(s).replace("sub-Neurotech", ""))

    demo["patient"] = demo["participant_id"].map(pid)
    findings["patient"] = findings["participant_id"].map(pid)
    cols_f = [c for c in findings.columns if c not in {"participant_id"}]
    merged = demo.merge(findings[cols_f], on="patient", how="left")
    return merged


def _metadata_fusion(val_df: pd.DataFrame, test_df: pd.DataFrame, pheno: pd.DataFrame) -> dict:
    """Second-stage logistic: model prob + clinical features → fused score."""
    feature_cols = []
    for c in ["age", "sex", "ever_abnormal", "any_epileptiform", "any_seizure"]:
        if c in pheno.columns:
            feature_cols.append(c)

    def featurize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        m = df.merge(pheno[["patient", *feature_cols]], on="patient", how="left")
        X_parts = [m["probability"].to_numpy(dtype=float).reshape(-1, 1)]
        for c in feature_cols:
            col = m[c]
            if col.dtype == object or str(col.dtype) == "string":
                # sex etc.
                mapped = col.astype(str).str.lower().map({"m": 1.0, "male": 1.0, "f": 0.0, "female": 0.0})
                X_parts.append(mapped.fillna(0.5).to_numpy(dtype=float).reshape(-1, 1))
            else:
                X_parts.append(pd.to_numeric(col, errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1))
        X = np.hstack(X_parts)
        y = m["label"].to_numpy(dtype=int)
        return X, y

    Xv, yv = featurize(val_df)
    Xt, yt = featurize(test_df)
    scaler = StandardScaler()
    Xv_s = scaler.fit_transform(Xv)
    Xt_s = scaler.transform(Xt)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xv_s, yv)
    pv = clf.predict_proba(Xv_s)[:, 1]
    pt = clf.predict_proba(Xt_s)[:, 1]
    val_out = val_df.copy()
    test_out = test_df.copy()
    val_out["probability"] = pv
    test_out["probability"] = pt
    val_out = _apply_max_smooth(val_out)
    test_out = _apply_max_smooth(test_out)
    return {
        "features": feature_cols + ["probability"],
        "val": _score(val_out, n_boot=50, tag="meta_val"),
        "test": _score(test_out, n_boot=50, tag="meta_test"),
    }


def _blend(dfs: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    names = [n for n in weights if n in dfs]
    key = ["record_id", "start_seconds"]
    base = dfs[names[0]][key + ["label", "duration_seconds", "patient"]].copy()
    base["probability"] = 0.0
    wsum = sum(weights[n] for n in names)
    for n in names:
        p = dfs[n][key + ["probability"]].rename(columns={"probability": f"p_{n}"})
        base = base.merge(p, on=key, how="inner")
        base["probability"] += (weights[n] / wsum) * base[f"p_{n}"]
    return base


def main() -> None:
    windows = load_windows()
    audit = _audit_splits(windows)
    (OUT / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("=== SPLIT AUDIT ===")
    print(json.dumps(audit, indent=2))

    results: dict = {"audit": audit, "models": {}}

    # Score available raw + max-smooth prediction CSVs
    for split_name, candidates in [("test", PRED_CANDIDATES), ("val", VAL_CANDIDATES)]:
        for name, path in candidates.items():
            df = _load_pred(path)
            if df is None:
                print(f"skip missing {split_name}/{name}")
                continue
            raw = _score(df, n_boot=100 if split_name == "test" else 50, tag=f"{name}_{split_name}_raw")
            results["models"][f"{name}_{split_name}_raw"] = raw
            print(
                f"{name} {split_name} raw: PR={raw['pr_auc']:.4f} PRG={raw['prg_auc']:.4f} "
                f"lift={raw['pr_lift']:.2f}x  bootPR=[{raw['boot_pr_auc_lo']:.3f},{raw['boot_pr_auc_hi']:.3f}] "
                f"events={raw['n_pos_events']:.0f} "
                f"P@R90={raw.get('precision_at_recall_r90', float('nan')):.3f} "
                f"FA24h@R90={raw.get('fp_per_24h_at_recall_r90', float('nan')):.1f}"
            )
            smooth = _apply_max_smooth(df)
            sm = _score(smooth, n_boot=100 if split_name == "test" else 50, tag=f"{name}_{split_name}_max15")
            results["models"][f"{name}_{split_name}_max15"] = sm
            print(
                f"{name} {split_name} max15: PR={sm['pr_auc']:.4f} PRG={sm['prg_auc']:.4f} "
                f"lift={sm['pr_lift']:.2f}x  bootPR=[{sm['boot_pr_auc_lo']:.3f},{sm['boot_pr_auc_hi']:.3f}] "
                f"event_sens={sm['event_sensitivity']:.3f} event_FA24h={sm['event_fa_per_24h']:.1f} "
                f"P@R90={sm.get('precision_at_recall_r90', float('nan')):.3f}"
            )

    # Ensemble ft + ft2 + s42 (equal weight) from raw CSVs, then max-smooth
    test_dfs = {k: _load_pred(v) for k, v in PRED_CANDIDATES.items()}
    val_dfs = {k: _load_pred(v) for k, v in VAL_CANDIDATES.items()}
    ens_names = [n for n in ["ft", "ft2", "s42"] if test_dfs.get(n) is not None]
    if len(ens_names) >= 2:
        # Prefer equal-weight 3-way blend (user-tune rarely beats this with ~600 val events).
        # Also report the val-PR-tuned weights for comparison.
        equal_w = {n: 1.0 / len(ens_names) for n in ens_names}
        best_w, best_pr = None, -1.0
        grid = [0.0, 0.2, 0.33, 0.5, 0.67, 0.8, 1.0]
        for wf in grid:
            for w2 in grid:
                w3 = 1.0 - wf - w2
                if w3 < -1e-9:
                    continue
                weights = {"ft": wf, "ft2": w2, "s42": max(w3, 0.0)}
                if any(n not in ens_names for n, w in weights.items() if w > 0):
                    continue
                if val_dfs.get("ft") is None:
                    break
                present = {n: val_dfs[n] for n in ens_names if val_dfs.get(n) is not None}
                for n, d in list(present.items()):
                    if "patient" not in d.columns:
                        wsplit = windows[windows["split"] == "val"][
                            ["record_id", "start_seconds", "patient"]
                        ]
                        present[n] = d.merge(wsplit, on=["record_id", "start_seconds"], how="left")
                blended = _blend(present, {n: weights[n] for n in present})
                blended = _apply_max_smooth(blended)
                pr = binary_metrics(
                    blended["label"].to_numpy(),
                    blended["probability"].to_numpy(),
                    float(blended["threshold"].iloc[0]),
                    float(blended["duration_seconds"].sum() / 3600),
                )["pr_auc"]
                if pr > best_pr:
                    best_pr = pr
                    best_w = {n: weights[n] for n in present}

        if best_w is None:
            best_w = equal_w

        print("=== ENSEMBLE equal weights ===", equal_w)
        print("=== ENSEMBLE val-tuned weights ===", best_w, "val_PR", best_pr)
        for label, weights in [("equal", equal_w), ("valtune", best_w)]:
            for split_name, dfs in [("val", val_dfs), ("test", test_dfs)]:
                present = {}
                for n in ens_names:
                    if dfs.get(n) is None:
                        continue
                    d = dfs[n]
                    if "patient" not in d.columns:
                        wsplit = windows[windows["split"] == split_name][
                            ["record_id", "start_seconds", "patient"]
                        ]
                        d = d.merge(wsplit, on=["record_id", "start_seconds"], how="left")
                    present[n] = d
                blended = _blend(present, {n: weights.get(n, 0.0) for n in present})
                blended = _apply_max_smooth(blended)
                met = _score(blended, n_boot=100, tag=f"ensemble3_{label}_{split_name}_max15")
                results["models"][f"ensemble3_{label}_{split_name}_max15"] = met
                blended.to_csv(
                    OUT / f"{split_name}_ensemble3_{label}_max15_predictions.csv", index=False
                )
                print(
                    f"ensemble3/{label} {split_name} max15: PR={met['pr_auc']:.4f} PRG={met['prg_auc']:.4f} "
                    f"lift={met['pr_lift']:.2f}x bootPR=[{met['boot_pr_auc_lo']:.3f},{met['boot_pr_auc_hi']:.3f}] "
                    f"P@R90={met.get('precision_at_recall_r90', float('nan')):.3f} "
                    f"event_FA24h={met['event_fa_per_24h']:.1f}"
                )
        results["ensemble_weights_equal"] = equal_w
        results["ensemble_weights_valtune"] = best_w

        # Skip the old single-weight block below
        if False:
            pass
        results["ensemble_weights"] = equal_w
        # Prevent falling through into the old loop body if any remains
        ens_names = []

    # Metadata fusion on top of best single (ft max-smooth)
    try:
        pheno = _load_phenotype()
        ft_val = _load_pred(VAL_CANDIDATES["ft"])
        ft_test = _load_pred(PRED_CANDIDATES["ft"])
        if ft_val is not None and ft_test is not None:
            for df, split in [(ft_val, "val"), (ft_test, "test")]:
                if "patient" not in df.columns:
                    wsplit = windows[windows["split"] == split][
                        ["record_id", "start_seconds", "patient"]
                    ]
                    if split == "val":
                        ft_val = df.merge(wsplit, on=["record_id", "start_seconds"], how="left")
                    else:
                        ft_test = df.merge(wsplit, on=["record_id", "start_seconds"], how="left")
            meta = _metadata_fusion(ft_val, ft_test, pheno)
            results["metadata_fusion"] = meta
            print(
                f"metadata fusion val: PR={meta['val']['pr_auc']:.4f} PRG={meta['val']['prg_auc']:.4f}"
            )
            print(
                f"metadata fusion test: PR={meta['test']['pr_auc']:.4f} PRG={meta['test']['prg_auc']:.4f} "
                f"P@R90={meta['test'].get('precision_at_recall_r90', float('nan')):.3f}"
            )
    except Exception as exc:  # noqa: BLE001
        results["metadata_fusion_error"] = str(exc)
        print("metadata fusion failed:", exc)

    # Optional GPU ensemble via engine if CSVs missing pieces
    missing_ckpts = [n for n in ["ft", "ft2", "s42"] if not PRED_CANDIDATES[n].exists() and CKPTS[n].exists()]
    if missing_ckpts:
        print("Running GPU ensemble for missing CSV models:", missing_ckpts)
        paths = [CKPTS[n] for n in ["ft", "ft2", "s42"] if CKPTS[n].exists()]
        if len(paths) >= 2:
            met = evaluate_ensemble(
                paths,
                windows,
                "test",
                OUT,
                smooth_seconds=15,
                smooth_mode="max",
                n_boot=50,
            )
            results["models"]["gpu_ensemble_test"] = met

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("Wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
