"""Phases 3 & 4 - choose a leakage-safe, class-balanced ~100 GB subset.

Design decisions that matter for the downstream seizure detector:

1. **Sample patients, not recordings.** The atomic unit is a patient; when a
   patient is chosen we take *all* their signal-bearing sessions. This mirrors
   the user's plan and prevents within-patient selection bias.

2. **No data leakage.** Train/val/test splits are assigned at the *patient*
   level, so no patient's recordings ever straddle two splits. Splits are
   stratified by whether the patient has any seizure recording.

3. **No class imbalance.** The raw corpus is ~29% seizure-positive by
   recording. We oversample seizure-positive patients toward a configurable
   byte-budget fraction (default 50%), so the detector sees plenty of positives
   while retaining genuine negatives.

4. **Respect the budget.** Patients are added greedily (shuffled, seeded) until
   the ~100 GB byte budget is met, filling a positive and a negative byte
   sub-budget separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import artifacts, config
from .config import SelectionConfig


def _patient_table(signal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signal recordings to one row per patient."""
    signal = signal.copy()
    signal["seizure_bytes"] = np.where(
        signal["is_seizure"], signal["size_bytes"], 0
    )
    signal["nonseizure_bytes"] = np.where(
        signal["is_seizure"], 0, signal["size_bytes"]
    )
    g = signal.groupby("patient")
    pt = g.agg(
        n_recordings=("record_id", "size"),
        total_bytes=("size_bytes", "sum"),
        seizure_bytes=("seizure_bytes", "sum"),
        nonseizure_bytes=("nonseizure_bytes", "sum"),
        n_seizure_recordings=("is_seizure", "sum"),
    ).reset_index()
    pt["is_positive"] = pt["n_seizure_recordings"] > 0
    return pt


def select_patients(
    labelled: pd.DataFrame | None = None,
    cfg: SelectionConfig = config.DEFAULT_SELECTION,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Pick patients under the byte budget with positive oversampling.

    Returns the selected *recordings* (all signal sessions of chosen patients)
    with an added ``patient_class`` column ("positive"/"negative").
    """
    if labelled is None:
        labelled = artifacts.load_df(config.LABELLED_MANIFEST_PATH)

    signal = labelled[labelled["has_signal"]].copy()
    signal["is_seizure"] = signal["is_seizure"].fillna(False).astype(bool)

    pt = _patient_table(signal)
    pt = pt[pt["n_recordings"] >= cfg.min_recordings_per_patient]

    pos = pt[pt["is_positive"]].sample(frac=1.0, random_state=cfg.seed)
    neg = pt[~pt["is_positive"]].sample(frac=1.0, random_state=cfg.seed + 1)

    # Budget is split between seizure-POSITIVE patients and seizure-FREE
    # patients. Whole-patient sampling means we can't hit an exact seizure-
    # *recording* fraction, but we can guarantee ~half the data comes from
    # patients who have seizures (heavy oversampling vs the ~29% base rate);
    # fine-grained class balance is then handled at the window level in
    # training.
    budget_pos = cfg.target_bytes * cfg.target_positive_fraction
    budget_neg = cfg.target_bytes * (1.0 - cfg.target_positive_fraction)

    chosen: list[int] = []
    filled_pos = 0  # bytes from seizure-positive patients
    filled_neg = 0  # bytes from seizure-free patients
    total = 0

    def _try_add(row, is_pos: bool) -> None:
        nonlocal filled_pos, filled_neg, total
        # Overflow guard: never exceed the total budget (unless nothing chosen
        # yet, so a single huge patient can't deadlock an empty selection).
        if chosen and total + row.total_bytes > cfg.target_bytes:
            return
        chosen.append(row.patient)
        total += row.total_bytes
        if is_pos:
            filled_pos += row.total_bytes
        else:
            filled_neg += row.total_bytes

    # Fill the positive-patient sub-budget.
    for row in pos.itertuples(index=False):
        if filled_pos >= budget_pos:
            break
        _try_add(row, is_pos=True)

    # Fill the seizure-free sub-budget.
    for row in neg.itertuples(index=False):
        if filled_neg >= budget_neg:
            break
        _try_add(row, is_pos=False)

    chosen_set = set(chosen)
    selected = signal[signal["patient"].isin(chosen_set)].copy()
    pt_class = pt.set_index("patient")["is_positive"]
    selected["patient_class"] = np.where(
        selected["patient"].map(pt_class), "positive", "negative"
    )
    selected = selected.sort_values(["patient", "session"]).reset_index(drop=True)

    if save:
        artifacts.save_df(selected, config.SELECTION_PATH)

    _print_selection_report(selected, cfg)
    return selected


def _print_selection_report(selected: pd.DataFrame, cfg: SelectionConfig) -> None:
    total = selected["size_bytes"].sum()
    pos_rec = selected[selected["is_seizure"]]
    pos_patient_rows = selected[selected["patient_class"] == "positive"]
    pos_patient_bytes = pos_patient_rows["size_bytes"].sum()
    print("  --- selection ---")
    print(f"  patients            : {selected['patient'].nunique():,}")
    print(f"    seizure-positive  : {pos_patient_rows['patient'].nunique():,}")
    print(f"  recordings          : {len(selected):,}")
    print(f"  seizure recordings  : {len(pos_rec):,} ({len(pos_rec)/max(len(selected),1):.1%})")
    print(f"  total size          : {total/1024**3:.2f} GB (budget {cfg.target_gb:.0f})")
    print(f"  positive-patient share: {pos_patient_bytes/max(total,1):.1%} "
          f"(target {cfg.target_positive_fraction:.0%})")


def select_all_patients(
    labelled: pd.DataFrame | None = None,
    *,
    annotated_only: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """Select every eligible patient/recording for full-scale training.

    Unlike ``select_patients`` there is no byte budget or positive oversampling:
    all annotated, signal-bearing recordings are included. Only recordings with
    a local or listed Xltek CSV are kept when ``annotated_only=True``, since the
    current weak-label window policy requires technician annotations.
    """
    if labelled is None:
        labelled = artifacts.load_df(config.LABELLED_MANIFEST_PATH)

    signal = labelled[labelled["has_signal"]].copy()
    signal["is_seizure"] = signal["is_seizure"].fillna(False).astype(bool)
    if annotated_only:
        signal = signal[signal["has_xltek"].fillna(False)]

    if signal.empty:
        raise ValueError("No annotated signal recordings found in labelled manifest")

    pt = _patient_table(signal)
    pt_class = pt.set_index("patient")["is_positive"]
    selected = signal.copy()
    selected["patient_class"] = np.where(
        selected["patient"].map(pt_class), "positive", "negative"
    )
    selected = selected.sort_values(["patient", "session"]).reset_index(drop=True)

    if save:
        artifacts.save_df(selected, config.SELECTION_PATH)

    total = selected["size_bytes"].sum()
    pos_rec = selected[selected["is_seizure"]]
    print("  --- full-scale selection ---")
    print(f"  patients            : {selected['patient'].nunique():,}")
    print(f"  recordings          : {len(selected):,}")
    print(f"  seizure recordings  : {len(pos_rec):,} ({len(pos_rec)/max(len(selected),1):.1%})")
    print(f"  total size          : {total/1024**4:.2f} TB")
    return selected


def make_splits(
    selected: pd.DataFrame | None = None,
    cfg: SelectionConfig = config.DEFAULT_SELECTION,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Assign patient-level train/val/test splits (stratified, leakage-safe)."""
    if selected is None:
        selected = artifacts.load_df(config.SELECTION_PATH)

    f_train, f_val, f_test = cfg.split_fractions
    if not np.isclose(f_train + f_val + f_test, 1.0):
        raise ValueError("split_fractions must sum to 1.0")

    # One (patient, is_positive) row per patient.
    pat = (
        selected.groupby("patient")["is_seizure"]
        .any()
        .rename("is_positive")
        .reset_index()
    )

    rng = np.random.default_rng(cfg.seed)
    assignments: dict[int, str] = {}
    # Stratify so each split has a comparable positive rate.
    for is_pos, grp in pat.groupby("is_positive"):
        ids = grp["patient"].to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * f_train))
        n_val = int(round(n * f_val))
        for pid in ids[:n_train]:
            assignments[pid] = "train"
        for pid in ids[n_train : n_train + n_val]:
            assignments[pid] = "val"
        for pid in ids[n_train + n_val :]:
            assignments[pid] = "test"

    out = selected.copy()
    out["split"] = out["patient"].map(assignments)

    if save:
        artifacts.save_df(out, config.SPLIT_PATH)

    _print_split_report(out)
    return out


def _print_split_report(out: pd.DataFrame) -> None:
    print("  --- splits (patient-level, no leakage) ---")
    for split in ("train", "val", "test"):
        s = out[out["split"] == split]
        if s.empty:
            continue
        pos = s[s["is_seizure"]]
        print(
            f"  {split:5s}: {s['patient'].nunique():4d} patients | "
            f"{len(s):5d} rec | {len(pos):5d} seizure ({len(pos)/max(len(s),1):.1%}) | "
            f"{s['size_bytes'].sum()/1024**3:6.2f} GB"
        )
    # Leakage assertion: patient sets must be disjoint across splits.
    per_split = {
        sp: set(out[out["split"] == sp]["patient"]) for sp in out["split"].unique()
    }
    splits = list(per_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = per_split[splits[i]] & per_split[splits[j]]
            assert not overlap, f"LEAKAGE: patients in both {splits[i]} & {splits[j]}"
    print("  leakage check       : OK (disjoint patients across splits)")
