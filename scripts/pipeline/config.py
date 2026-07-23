"""Configuration shared by the Neurotech acquisition pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

FILES_TXT = DATA_DIR / "files.txt"
FILES_TXT_ENCODING = "utf-16"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.parquet"
ANNOTATION_INDEX_PATH = ARTIFACTS_DIR / "annotation_index.parquet"
LABELLED_MANIFEST_PATH = ARTIFACTS_DIR / "manifest_labelled.parquet"
SELECTION_PATH = ARTIFACTS_DIR / "selected_recordings.parquet"
SPLIT_PATH = ARTIFACTS_DIR / "splits.parquet"

ACCESS_POINT = os.environ.get(
    "NEUROTECH_ACCESS_POINT",
    "bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias",
)
S3_PREFIX = "EEG/bids/Neurotech"
AWS_CLI = os.environ.get(
    "AWS_CLI_PATH",
    r"C:\Program Files\Amazon\AWSCLIV2\aws.exe" if os.name == "nt" else "aws",
)

MIN_SIGNAL_BYTES = 200 * 1024
EDF_SUFFIX = "_eeg.edf"
SIDECAR_SUFFIXES = {
    "json": "_eeg.json",
    "channels": "_channels.tsv",
    "xltek": "_Xltek.csv",
}

SEIZURE_KEYWORDS = ("seiz", " sz", "ictal", "clonic", "convuls")
SPIKE_KEYWORDS = ("spike", "spk", "polyspike")
SHARP_KEYWORDS = ("sharp", "sw ", "sharp-wave", "sharp wave")
NEGATION_PREFIXES = ("no ", "non-", "non ", "without ", "denies ", "-free")


@dataclass
class SelectionConfig:
    target_gb: float = 100.0
    target_positive_fraction: float = 0.5
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    min_recordings_per_patient: int = 1
    seed: int = 1337

    @property
    def target_bytes(self) -> int:
        return int(self.target_gb * 1024**3)


@dataclass
class DownloadConfig:
    max_workers: int = 8
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    max_retry_rounds: int = 100
    required_sidecars: tuple[str, ...] = ("json", "channels")
    optional_sidecars: tuple[str, ...] = ("xltek",)


DEFAULT_SELECTION = SelectionConfig()
DEFAULT_DOWNLOAD = DownloadConfig()

METADATA_FILES = (
    "README",
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
)
PHENOTYPE_DIR = "phenotype"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
