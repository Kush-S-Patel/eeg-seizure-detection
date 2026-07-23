"""Download the selected EDF recordings (and their sidecars).

Runs after selection/splitting. Idempotent and resumable: existing non-empty
files are skipped, so re-running only fetches what's missing. Downloads run in a
thread pool since each object is an independent ``aws s3 cp`` subprocess (I/O
bound).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import artifacts, awscli, config
from .config import DownloadConfig


def _keys_for_record(row, dl: DownloadConfig) -> list[tuple[str, bool]]:
    """(key, required) list for one recording: EDF + sidecars that exist.

    ``row`` is a manifest row (namedtuple/Series-like) carrying ``record_id``
    and ``has_<sidecar>`` flags derived from the S3 listing. We only request a
    sidecar when the listing says it exists, so absent files (e.g. the ~9k
    recordings without an Xltek CSV) never trigger a 404.
    """
    record_id = row.record_id
    items: list[tuple[str, bool]] = [(record_id + config.EDF_SUFFIX, True)]

    def _sidecar_exists(name: str) -> bool:
        flag = getattr(row, f"has_{name}", None)
        # If the flag is missing (older artifact), attempt the download anyway.
        return True if flag is None else bool(flag)

    for name in dl.required_sidecars:
        if _sidecar_exists(name):
            items.append((record_id + config.SIDECAR_SUFFIXES[name], True))
    for name in dl.optional_sidecars:
        if _sidecar_exists(name):
            items.append((record_id + config.SIDECAR_SUFFIXES[name], False))
    return items


def _download_record(row, dl: DownloadConfig) -> bool:
    """Download one recording's files. Returns False iff a required file fails."""
    ok = True
    for key, required in _keys_for_record(row, dl):
        local = config.RAW_DIR / key
        expected_size = (
            int(row.size_bytes) if key.endswith(config.EDF_SUFFIX) else None
        )
        success = awscli.cp(
            key,
            local,
            max_retries=dl.max_retries,
            backoff_s=dl.retry_backoff_s,
            warn_on_missing=required,
            expected_size_bytes=expected_size,
        )
        if not success and required:
            ok = False
    return ok


def _download_rows(rows: list, dl: DownloadConfig) -> tuple[int, list]:
    """Download a list of recording rows. Returns (ok_count, failed_rows)."""
    n = len(rows)
    if not n:
        return 0, []
    done = failed = 0
    failed_rows: list = []
    with ThreadPoolExecutor(max_workers=dl.max_workers) as pool:
        futures = {pool.submit(_download_record, row, dl): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            if fut.result():
                done += 1
            else:
                failed += 1
                failed_rows.append(row)
                print(f"    ! required file failed for {row.record_id}", flush=True)
            if i % 25 == 0 or i == n:
                print(f"  progress: {i}/{n} (ok={done}, failed={failed})", flush=True)
    return done, failed_rows


def download_selection(
    selection: pd.DataFrame | None = None,
    dl: DownloadConfig = config.DEFAULT_DOWNLOAD,
    *,
    splits: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> dict:
    """Download every selected recording in parallel.

    Prefers the split table (so ``splits`` filtering works); falls back to the
    raw selection. Pass ``splits=("train",)`` to fetch only certain splits.
    """
    if selection is None:
        path = (
            config.SPLIT_PATH
            if artifacts.exists(config.SPLIT_PATH)
            else config.SELECTION_PATH
        )
        selection = artifacts.load_df(path)

    if splits is not None and "split" in selection.columns:
        selection = selection[selection["split"].isin(splits)]

    total_gb = selection["size_bytes"].sum() / 1024 ** 3
    rows = list(selection.itertuples(index=False))
    n = len(rows)
    print(
        f"  {n:,} recordings, {total_gb:.2f} GB to fetch "
        f"(workers={dl.max_workers})",
        flush=True,
    )

    if dry_run:
        print("  dry-run: nothing downloaded.")
        return {"records": n, "downloaded": 0, "failed": 0, "gb": total_gb}

    config.ensure_dirs()
    total_done = 0
    pending = list(rows)
    round_num = 0
    while pending:
        round_num += 1
        if round_num > 1:
            print(
                f"  retry round {round_num - 1}: {len(pending):,} recordings still pending",
                flush=True,
            )
        done, pending = _download_rows(pending, dl)
        total_done += done
        if not pending:
            break
        if round_num >= dl.max_retry_rounds:
            print(
                f"  giving up after {round_num} rounds; {len(pending):,} recordings still failed",
                flush=True,
            )
            break

    failed = len(pending)
    print("  --- download complete ---", flush=True)
    print(f"  recordings ok : {total_done:,}", flush=True)
    print(f"  recordings bad: {failed:,}", flush=True)
    return {"records": n, "downloaded": total_done, "failed": failed, "gb": total_gb}
