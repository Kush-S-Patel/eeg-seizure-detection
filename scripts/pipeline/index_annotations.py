"""Parse locally-synced technician annotation CSVs into recording labels.

Produces a labelled manifest with per-recording event counts and a boolean
``is_seizure`` flag, which the selection stage uses to oversample positives.

The Xltek CSV schema is not published, so parsing is deliberately
format-agnostic: every cell is scanned as free text for keyword matches, with a
simple negation guard ("no seizures", "seizure-free", ...). Refine the keyword
sets in ``config`` once a few real CSVs have been inspected.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from . import artifacts, config


def _line_has(text: str, keywords) -> bool:
    return any(kw in text for kw in keywords)


def _is_negated(text: str) -> bool:
    return any(neg in text for neg in config.NEGATION_PREFIXES)


def parse_csv(path: Path) -> dict:
    """Return event counts for one annotation CSV.

    Counts *rows* (annotation events) whose text matches each category.
    """
    counts = {"n_annotations": 0, "n_seizure": 0, "n_spike": 0, "n_sharp": 0}
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            for i, row in enumerate(reader):
                if i == 0 and any(
                    h.strip().lower() in {"onset", "time", "label", "annotation"}
                    for h in row
                ):
                    continue  # header row
                text = " ".join(row).lower()
                if not text.strip():
                    continue
                counts["n_annotations"] += 1
                if _is_negated(text):
                    continue
                if _line_has(text, config.SEIZURE_KEYWORDS):
                    counts["n_seizure"] += 1
                if _line_has(text, config.SPIKE_KEYWORDS):
                    counts["n_spike"] += 1
                if _line_has(text, config.SHARP_KEYWORDS):
                    counts["n_sharp"] += 1
    except FileNotFoundError:
        pass
    return counts


def build_index(manifest: pd.DataFrame | None = None, *, save: bool = True) -> pd.DataFrame:
    """Join parsed annotation counts onto the manifest.

    Adds columns: n_annotations, n_seizure, n_spike, n_sharp, has_labels,
    is_seizure. Recordings without a local CSV get has_labels=False and are
    treated as negatives during selection.
    """
    if manifest is None:
        manifest = artifacts.load_df(config.MANIFEST_PATH)

    records = []
    parsed = 0
    for row in manifest.itertuples(index=False):
        rec = {"record_id": row.record_id}
        if getattr(row, "has_xltek", False):
            csv_path = config.RAW_DIR / (row.record_id + config.SIDECAR_SUFFIXES["xltek"])
            if csv_path.exists():
                rec.update(parse_csv(csv_path))
                rec["has_labels"] = True
                parsed += 1
            else:
                rec["has_labels"] = False
        else:
            rec["has_labels"] = False
        records.append(rec)

    idx = pd.DataFrame(records)
    for col in ("n_annotations", "n_seizure", "n_spike", "n_sharp"):
        if col not in idx:
            idx[col] = 0
        idx[col] = idx[col].fillna(0).astype(int)
    idx["has_labels"] = idx["has_labels"].fillna(False)
    idx["is_seizure"] = idx["n_seizure"] > 0

    print(f"  parsed {parsed:,} annotation CSVs")
    if parsed == 0:
        print("  ! no local CSVs found - run the 'annotations' stage first.")

    labelled = manifest.merge(idx, on="record_id", how="left")

    if save:
        artifacts.save_df(labelled, config.LABELLED_MANIFEST_PATH)
    return labelled


def summarize(labelled: pd.DataFrame) -> str:
    signal = labelled[labelled["has_signal"]]
    pos = signal[signal["is_seizure"]]
    return "\n".join(
        [
            f"Signal recordings        : {len(signal):,}",
            f"  with parsed labels     : {int(signal['has_labels'].sum()):,}",
            f"  seizure-positive       : {len(pos):,} ({len(pos) / max(len(signal),1):.1%})",
            f"  total seizure events   : {int(signal['n_seizure'].sum()):,}",
            f"  total spike events     : {int(signal['n_spike'].sum()):,}",
            f"  total sharp events     : {int(signal['n_sharp'].sum()):,}",
            f"Seizure-positive patients: {pos['patient'].nunique():,}",
        ]
    )
