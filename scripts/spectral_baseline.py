"""Spectral logistic-regression baseline for seizure-marker-proximity detection.

Fits on a stratified train subsample; evaluates on the full test split using
bandpower (+ optional mean PLV) features from the window cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seizure_detector.cache import load_window_cache
from seizure_detector.config import WINDOW_CACHE_DIR, WINDOWS_PATH
from seizure_detector.features import bandpower_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "showcase" / "metrics" / "spectral_lr_test.json"


def _features(window_ids: np.ndarray, x_mm, mask_mm, batch: int = 128) -> np.ndarray:
    chunks = []
    for i in range(0, len(window_ids), batch):
        ids = window_ids[i : i + batch]
        xb = torch.from_numpy(np.array(x_mm[ids]))
        mb = torch.from_numpy(np.array(mask_mm[ids]))
        xb = xb * mb.unsqueeze(-1)
        with torch.no_grad():
            bp = bandpower_features(xb).numpy()
        chunks.append(bp)
    return np.concatenate(chunks, axis=0)


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_pos = min(len(pos), max(1, int(0.2 * n)))
    n_neg = min(len(neg), n - n_pos)
    rng = np.random.default_rng(seed)
    parts = [
        pos.sample(n_pos, random_state=seed) if len(pos) > n_pos else pos,
        neg.sample(n_neg, random_state=seed + 1) if len(neg) > n_neg else neg,
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(
    train_n: int = 40_000,
    test_n: int | None = None,
    seed: int = 1337,
) -> None:
    windows = pd.read_parquet(WINDOWS_PATH)
    x_mm, mask_mm, meta = load_window_cache(WINDOW_CACHE_DIR)
    print(f"cache n_rows={meta['n_rows']:,}")

    train = _subsample(windows[windows["split"] == "train"], train_n, seed)
    test = windows[windows["split"] == "test"].reset_index(drop=True)
    if test_n is not None:
        test = _subsample(test, test_n, seed + 7)

    print(f"train subsample={len(train):,} prev={train.label.mean():.4f}")
    print(f"test n={len(test):,} prev={test.label.mean():.4f}")

    X_train = _features(train["window_id"].to_numpy(dtype=int), x_mm, mask_mm)
    y_train = train["label"].to_numpy(dtype=int)
    X_test = _features(test["window_id"].to_numpy(dtype=int), x_mm, mask_mm)
    y_test = test["label"].to_numpy(dtype=int)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pr = float(average_precision_score(y_test, proba))
    roc = float(roc_auc_score(y_test, proba))
    payload = {
        "model": "logistic_regression_bandpower",
        "description": (
            "StandardScaler + LogisticRegression(class_weight=balanced) on "
            "per-window bandpower features (5 bands × 18 channels)."
        ),
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "test_prevalence": float(y_test.mean()),
        "pr_auc": pr,
        "roc_auc": roc,
        "pr_lift": float(pr / max(y_test.mean(), 1e-12)),
        "seed": seed,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
