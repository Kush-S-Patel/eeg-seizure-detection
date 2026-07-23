"""Validate local selected data before preparation or evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, RAW_DIR, SPLITS_PATH
from .paths import load_splits, recording_paths


def _read_edf_header(path: Path) -> tuple[bool, str]:
    try:
        import mne

        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        if raw.n_times <= 0 or raw.info["sfreq"] <= 0:
            return False, "empty EDF"
        return True, f"{raw.n_times / raw.info['sfreq']:.1f}s, {len(raw.ch_names)}ch"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def audit_dataset(
    splits_path: Path = SPLITS_PATH,
    raw_dir: Path = RAW_DIR,
    *,
    read_headers: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in load_splits(splits_path).itertuples(index=False):
        paths = recording_paths(rec.record_id, raw_dir)
        edf_exists = paths["edf"].is_file() and paths["edf"].stat().st_size > 0
        header_ok, detail = (False, "missing")
        if edf_exists:
            header_ok, detail = _read_edf_header(paths["edf"]) if read_headers else (True, "not read")
        rows.append(
            {
                "record_id": rec.record_id,
                "patient": rec.patient,
                "session": rec.session,
                "split": rec.split,
                "edf_exists": edf_exists,
                "header_ok": header_ok,
                "json_exists": paths["json"].is_file(),
                "channels_exists": paths["channels"].is_file(),
                "xltek_exists": paths["xltek"].is_file(),
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def print_report(report: pd.DataFrame) -> bool:
    print("split  records  edf  readable  json  channels  annotations")
    for split, part in report.groupby("split", sort=False):
        print(
            f"{split:5}  {len(part):7d}  {part.edf_exists.sum():3d}  "
            f"{part.header_ok.sum():8d}  {part.json_exists.sum():4d}  "
            f"{part.channels_exists.sum():8d}  {part.xltek_exists.sum():11d}"
        )
    failures = report[~report["header_ok"]]
    if len(failures):
        print(f"\nMissing/unreadable EDFs: {len(failures)}")
        print(failures[["split", "record_id", "detail"]].head(20).to_string(index=False))
    leakage = report.groupby("patient")["split"].nunique().gt(1).any()
    print(f"\nPatient leakage: {'DETECTED' if leakage else 'none'}")
    return failures.empty and not leakage


def save_report(report: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ARTIFACTS_DIR / "data_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return path
