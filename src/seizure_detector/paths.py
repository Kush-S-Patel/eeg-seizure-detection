"""Artifact and BIDS path resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DIR, SPLITS_PATH


def load_table(path: Path) -> pd.DataFrame:
    """Load Parquet, falling back to the same path with a CSV suffix."""
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing {path} and {csv_path}")


def load_splits(path: Path = SPLITS_PATH) -> pd.DataFrame:
    table = load_table(path)
    required = {"record_id", "patient", "session", "split"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Split table is missing columns: {sorted(missing)}")
    overlaps = table.groupby("patient")["split"].nunique()
    if (overlaps > 1).any():
        bad = overlaps[overlaps > 1].index.tolist()
        raise ValueError(f"Patient leakage in split table: {bad[:10]}")
    return table


def recording_paths(record_id: str, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    base = raw_dir / record_id
    return {
        "edf": Path(f"{base}_eeg.edf"),
        "json": Path(f"{base}_eeg.json"),
        "channels": Path(f"{base}_channels.tsv"),
        "xltek": Path(f"{base}_Xltek.csv"),
    }
