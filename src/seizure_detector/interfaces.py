"""Stable extension seams for labels, preprocessing, models, and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class LabelPolicy(Protocol):
    def seizure_times(self, annotation_path: Path, recording_start) -> np.ndarray: ...


class Preprocessor(Protocol):
    def transform(self, edf_path: Path, start_seconds: float, duration_seconds: float) -> np.ndarray: ...


class MetricHook(Protocol):
    def __call__(self, targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]: ...


# IMPROVEMENT: implement these protocols for expert interval labels, artifact
# rejection, alternate montages, or site-specific metrics without changing the
# dataset/training loops.
