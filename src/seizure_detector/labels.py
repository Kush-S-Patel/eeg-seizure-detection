"""Weak-label parsing for technician Xltek event CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEIZURE_RE = re.compile(r"\b(seiz\w*|ictal|convuls\w*|clonic|\bsz\b)", re.IGNORECASE)
NEGATED_RE = re.compile(
    r"\b(no|without|denies|non)[ -]{0,2}(seiz\w*|ictal|convuls\w*|clonic|\bsz\b)"
    r"|seizure[- ]free",
    re.IGNORECASE,
)


def is_seizure_text(text: str) -> bool:
    return bool(SEIZURE_RE.search(text or "")) and not bool(NEGATED_RE.search(text or ""))


def _naive_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def read_seizure_times(annotation_path: Path, recording_start) -> np.ndarray:
    """Return seizure marker offsets in seconds from EDF start.

    Xltek only supplies point timestamps, not expert-reviewed onset/duration
    intervals. These offsets are therefore weak labels, not ground truth.
    """
    if not annotation_path.is_file() or recording_start is None:
        return np.empty(0, dtype=np.float64)
    table = pd.read_csv(annotation_path, encoding_errors="replace")
    if not {"Text", "CreationTime"}.issubset(table.columns):
        return np.empty(0, dtype=np.float64)
    start = _naive_timestamp(recording_start)
    selected = table[table["Text"].fillna("").map(is_seizure_text)].copy()
    times = pd.to_datetime(selected["CreationTime"], errors="coerce")
    offsets: list[float] = []
    for value in times.dropna():
        offsets.append((_naive_timestamp(value) - start).total_seconds())
    return np.asarray(sorted(set(offsets)), dtype=np.float64)
