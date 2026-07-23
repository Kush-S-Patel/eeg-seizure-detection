"""Seizure forecasting (preictal) labels and chance baselines.

Re-labels existing detection windows using approximate onset times derived from
detection-positive clusters (Xltek point markers ±30s). No EDF rebuild required.

Horizon convention (AES / NeuroVista style):
  - SPH (seizure prediction horizon): minimum lead time before onset
  - SOP (seizure occurrence period): length of the preictal alarm window
  Preictal positives: centers in ``[onset - SOP - SPH, onset - SPH)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, WINDOWS_PATH

FORECAST_WINDOWS_PATH = ARTIFACTS_DIR / "windows_forecast.parquet"
ONSETS_PATH = ARTIFACTS_DIR / "seizure_onsets.parquet"


@dataclass(frozen=True)
class ForecastConfig:
    """Preictal / interictal labeling policy."""

    sop_seconds: float = 1800.0  # 30 min occurrence period
    sph_seconds: float = 300.0  # 5 min prediction horizon
    peri_ictal_guard_seconds: float = 300.0  # exclude ±5 min around onset
    post_ictal_guard_seconds: float = 1800.0  # 30 min after onset
    interictal_guard_seconds: float = 7200.0  # 2 h from any onset
    max_negatives_per_recording: int = 200
    max_positives_per_recording: int | None = None
    onset_cluster_gap_seconds: float = 60.0
    seed: int = 1337
    include_seizure_free_negatives: bool = True
    # Cleaner clinical subsets (help when weak onsets + dense clusters muddy labels).
    first_onset_only: bool = False
    min_inter_onset_gap_seconds: float = 0.0  # drop onsets closer than this to a prior onset

    def to_dict(self) -> dict:
        return asdict(self)


def filter_onsets(onsets: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """Optionally keep isolated / first-only onsets for cleaner preictal labels."""
    onsets = np.sort(np.asarray(onsets, dtype=float))
    if not len(onsets):
        return onsets
    if cfg.min_inter_onset_gap_seconds > 0:
        kept = [float(onsets[0])]
        for onset in onsets[1:]:
            if float(onset) - kept[-1] >= cfg.min_inter_onset_gap_seconds:
                kept.append(float(onset))
        onsets = np.asarray(kept, dtype=np.float64)
    if cfg.first_onset_only and len(onsets):
        onsets = onsets[:1]
    return onsets


def estimate_onsets_from_detection_windows(
    starts: np.ndarray,
    labels: np.ndarray,
    *,
    window_seconds: float = 10.0,
    cluster_gap_seconds: float = 60.0,
) -> np.ndarray:
    """Approximate seizure onsets as medians of clustered detection-positive centers."""
    starts = np.asarray(starts, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = starts[labels == 1]
    if not len(pos):
        return np.empty(0, dtype=np.float64)
    pos = np.sort(pos)
    centers = pos + window_seconds / 2.0
    onsets: list[float] = []
    cluster = [centers[0]]
    for c in centers[1:]:
        if c - cluster[-1] <= cluster_gap_seconds:
            cluster.append(c)
        else:
            onsets.append(float(np.median(cluster)))
            cluster = [c]
    onsets.append(float(np.median(cluster)))
    return np.asarray(onsets, dtype=np.float64)


def label_forecast_window(
    center: float,
    onsets: np.ndarray,
    cfg: ForecastConfig,
) -> int | None:
    """Return 1=preictal, 0=interictal, None=exclude (peri/post-ictal / ambiguous)."""
    onsets = np.asarray(onsets, dtype=float)
    if not len(onsets):
        return 0  # seizure-free recording → interictal

    # Distance to nearest past / future onset.
    future = onsets[onsets > center]
    past = onsets[onsets <= center]
    if len(future):
        ttn = float(future.min() - center)
    else:
        ttn = float("inf")
    if len(past):
        tts = float(center - past.max())  # time since last seizure
    else:
        tts = float("inf")

    # Peri-ictal / post-ictal exclusion zones.
    if ttn < cfg.peri_ictal_guard_seconds:
        return None
    if tts < cfg.post_ictal_guard_seconds:
        return None

    preictal_lo = cfg.sph_seconds
    preictal_hi = cfg.sph_seconds + cfg.sop_seconds
    if preictal_lo <= ttn < preictal_hi:
        return 1

    # Interictal: far from any upcoming seizure and past seizure.
    if ttn >= cfg.interictal_guard_seconds and tts >= cfg.interictal_guard_seconds:
        return 0
    return None


def build_onset_table(
    detection_windows: pd.DataFrame,
    *,
    cfg: ForecastConfig = ForecastConfig(),
    output_path: Path = ONSETS_PATH,
) -> pd.DataFrame:
    rows = []
    for record_id, group in detection_windows.groupby("record_id", sort=False):
        onsets = filter_onsets(
            estimate_onsets_from_detection_windows(
                group["start_seconds"].to_numpy(),
                group["label"].to_numpy(),
                window_seconds=float(group["duration_seconds"].iloc[0]),
                cluster_gap_seconds=cfg.onset_cluster_gap_seconds,
            ),
            cfg,
        )
        n_mark = int(group["n_recording_seizure_markers"].iloc[0])
        for onset in onsets:
            rows.append(
                {
                    "record_id": record_id,
                    "patient": int(group["patient"].iloc[0]),
                    "split": group["split"].iloc[0],
                    "onset_seconds": float(onset),
                    "n_recording_seizure_markers": n_mark,
                    "n_estimated_onsets": int(len(onsets)),
                }
            )
        if not len(onsets) and n_mark == 0:
            rows.append(
                {
                    "record_id": record_id,
                    "patient": int(group["patient"].iloc[0]),
                    "split": group["split"].iloc[0],
                    "onset_seconds": float("nan"),
                    "n_recording_seizure_markers": 0,
                    "n_estimated_onsets": 0,
                }
            )
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    return table


def build_forecast_windows(
    detection_windows: pd.DataFrame | None = None,
    *,
    cfg: ForecastConfig = ForecastConfig(),
    detection_path: Path = WINDOWS_PATH,
    output_path: Path = FORECAST_WINDOWS_PATH,
) -> pd.DataFrame:
    """Re-label cached detection windows for preictal forecasting."""
    if detection_windows is None:
        detection_windows = pd.read_parquet(detection_path)
    rng = np.random.default_rng(cfg.seed)
    onset_map: dict[str, np.ndarray] = {}
    for record_id, group in detection_windows.groupby("record_id", sort=False):
        raw = estimate_onsets_from_detection_windows(
            group["start_seconds"].to_numpy(),
            group["label"].to_numpy(),
            window_seconds=float(group["duration_seconds"].iloc[0]),
            cluster_gap_seconds=cfg.onset_cluster_gap_seconds,
        )
        onset_map[str(record_id)] = filter_onsets(raw, cfg)

    rows: list[dict] = []
    for record_id, group in detection_windows.groupby("record_id", sort=False):
        onsets = onset_map[str(record_id)]
        if not len(onsets) and not cfg.include_seizure_free_negatives:
            continue
        pos_idx: list[int] = []
        neg_idx: list[int] = []
        centers = group["start_seconds"].to_numpy(dtype=float) + group["duration_seconds"].to_numpy(
            dtype=float
        ) / 2.0
        for i, center in enumerate(centers):
            lab = label_forecast_window(float(center), onsets, cfg)
            if lab == 1:
                pos_idx.append(i)
            elif lab == 0:
                neg_idx.append(i)
        if cfg.max_positives_per_recording is not None and len(pos_idx) > cfg.max_positives_per_recording:
            pos_idx = list(rng.choice(pos_idx, cfg.max_positives_per_recording, replace=False))
        if len(neg_idx) > cfg.max_negatives_per_recording:
            neg_idx = list(rng.choice(neg_idx, cfg.max_negatives_per_recording, replace=False))
        chosen = pos_idx + neg_idx
        labels = [1] * len(pos_idx) + [0] * len(neg_idx)
        for index, label in zip(chosen, labels):
            row = group.iloc[int(index)].to_dict()
            row["label"] = int(label)
            row["label_source"] = "preictal_sop_sph"
            row["time_to_next_seizure"] = (
                float(onsets[onsets > centers[index]].min() - centers[index])
                if len(onsets) and np.any(onsets > centers[index])
                else float("nan")
            )
            row["n_estimated_onsets"] = int(len(onsets))
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No forecast windows produced; check SOP/SPH / guards")
    # Preserve cache indexing.
    if "window_id" not in table.columns:
        raise ValueError("detection windows missing window_id")
    table = table.reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    (output_path.with_suffix(".json")).write_text(
        __import__("json").dumps(
            {
                "config": cfg.to_dict(),
                "n_windows": int(len(table)),
                "n_pos": int(table["label"].sum()),
                "n_neg": int((table["label"] == 0).sum()),
                "prevalence": float(table["label"].mean()),
                "n_records": int(table["record_id"].nunique()),
                "n_patients": int(table["patient"].nunique()),
                "by_split": table.groupby("split")["label"]
                .agg(["count", "sum"])
                .rename(columns={"count": "n", "sum": "n_pos"})
                .to_dict(orient="index"),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    return table


def permute_onsets_within_record(
    onsets: np.ndarray,
    *,
    recording_duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular-shift onset times within a recording (chance baseline)."""
    if not len(onsets) or recording_duration <= 0:
        return onsets
    shift = float(rng.uniform(0, recording_duration))
    return np.sort((onsets + shift) % recording_duration)


def chance_label_shuffle(labels: pd.Series, seed: int = 1337) -> pd.Series:
    """Permute labels while preserving class counts (naive chance)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(labels.to_numpy()), index=labels.index, name=labels.name)
