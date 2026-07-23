"""Phase 2 - build a per-recording manifest from the raw S3 listing.

Parses ``data/files.txt`` (the output of
``aws s3 ls s3://.../EEG/bids/Neurotech/ --recursive``) into one row per EEG
recording, capturing patient, session, signal size, and which sidecar files
exist. This is the backbone every later stage joins against.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import artifacts, config

# sub-Neurotech<digits> ... ses-<digits>
_PATIENT_RE = re.compile(r"sub-Neurotech(\d+)")
_SESSION_RE = re.compile(r"ses-(\d+)")


def _iter_listing(files_txt: Path):
    """Yield (size_bytes, key) tuples from the `s3 ls --recursive` dump."""
    with open(files_txt, "r", encoding=config.FILES_TXT_ENCODING) as fh:
        for line in fh:
            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                continue
            _date, _time, size_str, key = parts
            try:
                size = int(size_str)
            except ValueError:
                continue
            yield size, key.strip()


def _record_base(key: str) -> str | None:
    """Return the shared filename stem for an EDF key, else None.

    e.g. ".../sub-Neurotech1_ses-1_task-EEG_eeg.edf"
      -> ".../sub-Neurotech1_ses-1_task-EEG"
    """
    if key.endswith(config.EDF_SUFFIX):
        return key[: -len(config.EDF_SUFFIX)]
    return None


def build_manifest(
    files_txt: Path | None = None,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Parse the listing into a recording-level manifest.

    Columns
    -------
    record_id      : filename stem shared by an EDF and its sidecars
    patient        : integer patient id (from sub-Neurotech<N>)
    session        : integer session id (from ses-<N>)
    edf_key        : S3 key of the EDF
    size_bytes     : EDF size
    has_signal     : size >= MIN_SIGNAL_BYTES (filters header-only stubs)
    has_json/has_channels/has_xltek : sidecar presence flags
    """
    files_txt = Path(files_txt or config.FILES_TXT)
    if not files_txt.exists():
        raise FileNotFoundError(
            f"{files_txt} not found. Generate it with:\n"
            f'  aws s3 ls s3://{config.ACCESS_POINT}/{config.S3_PREFIX}/ '
            f"--recursive > files.txt"
        )

    # First pass: collect EDF rows and a set of all keys for sidecar lookup.
    all_keys: set[str] = set()
    edf_rows: list[dict] = []

    for size, key in _iter_listing(files_txt):
        all_keys.add(key)
        base = _record_base(key)
        if base is None:
            continue
        p = _PATIENT_RE.search(key)
        s = _SESSION_RE.search(key)
        if not (p and s):
            continue
        edf_rows.append(
            {
                "record_id": base,
                "patient": int(p.group(1)),
                "session": int(s.group(1)),
                "edf_key": key,
                "size_bytes": size,
            }
        )

    df = pd.DataFrame(edf_rows)
    if df.empty:
        raise ValueError("No EDF entries parsed from listing; check the file.")

    df["has_signal"] = df["size_bytes"] >= config.MIN_SIGNAL_BYTES
    for name, suffix in config.SIDECAR_SUFFIXES.items():
        df[f"has_{name}"] = (df["record_id"] + suffix).isin(all_keys)

    df = df.sort_values(["patient", "session"]).reset_index(drop=True)

    if save:
        config.ensure_dirs()
        artifacts.save_df(df, config.MANIFEST_PATH)

    return df


def summarize(df: pd.DataFrame) -> str:
    signal = df[df["has_signal"]]
    lines = [
        f"Total EDF entries      : {len(df):,}",
        f"Signal-bearing (>= floor): {len(signal):,}",
        f"Header-only stubs      : {len(df) - len(signal):,}",
        f"Unique patients        : {df['patient'].nunique():,}",
        f"Patients w/ signal     : {signal['patient'].nunique():,}",
        f"Signal-bearing w/ Xltek: {int(signal['has_xltek'].sum()):,}",
        f"Total signal size      : {signal['size_bytes'].sum() / 1024**4:.2f} TB",
    ]
    return "\n".join(lines)
