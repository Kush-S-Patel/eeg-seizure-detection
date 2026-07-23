"""Neurotech EEG 100 GB acquisition pipeline (CLI orchestrator).

Turns the raw S3 listing (``data/files.txt``) into a leakage-safe,
class-balanced ~100 GB subset ready for training a seizure detector.

The old version of this script downloaded EDFs in *file order* until it hit
100 GB. That is unsafe for ML: it puts the same patient in train and test
(data leakage) and gives no control over the seizure/non-seizure balance. This
version runs a staged pipeline instead.

Stages (run individually or via `all`):

    manifest      Parse files.txt -> per-recording manifest.
    metadata      Download cohort metadata (participants, phenotype/, README).
    annotations   Sync every technician annotation CSV (tiny) for labelling.
    index         Parse annotation CSVs -> per-recording seizure labels.
    select        Patient-level sampling + positive oversampling under budget.
    split         Patient-level train/val/test split (no leakage).
    download      Download the selected EDF recordings (parallel, resumable).
    fullscale     Full dataset: header manifest + rolling cache (EC2-scale).
    all           Run every stage end to end.

Examples
--------
    # Prepare everything up to (but not including) the big download:
    python scripts/downloader_100gb.py all --skip-download

    # Then fetch the signals (e.g. only the training split first):
    python scripts/downloader_100gb.py download --splits train

    # Tune the subset:
    python scripts/downloader_100gb.py select --target-gb 100 --positive-fraction 0.5

Note: the download stages require the AWS CLI, valid credentials, and an
accepted BDSP data-use agreement. All planning stages (manifest/index/select/
split) run offline from files.txt + the synced annotation CSVs.
"""

from __future__ import annotations

import argparse
import sys

from pipeline import config
from pipeline.config import DownloadConfig, SelectionConfig


def _selection_cfg(args) -> SelectionConfig:
    base = config.DEFAULT_SELECTION
    return SelectionConfig(
        target_gb=getattr(args, "target_gb", None) or base.target_gb,
        target_positive_fraction=(
            getattr(args, "positive_fraction", None)
            if getattr(args, "positive_fraction", None) is not None
            else base.target_positive_fraction
        ),
        split_fractions=base.split_fractions,
        min_recordings_per_patient=base.min_recordings_per_patient,
        seed=getattr(args, "seed", None) or base.seed,
    )


def _download_cfg(args) -> DownloadConfig:
    base = config.DEFAULT_DOWNLOAD
    return DownloadConfig(
        max_workers=getattr(args, "workers", None) or base.max_workers,
        max_retries=base.max_retries,
        retry_backoff_s=base.retry_backoff_s,
        required_sidecars=base.required_sidecars,
        optional_sidecars=base.optional_sidecars,
    )


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #
def stage_manifest(args):
    from pipeline import manifest

    print("[manifest] parsing", config.FILES_TXT)
    df = manifest.build_manifest()
    print(manifest.summarize(df))


def stage_metadata(args):
    from pipeline import metadata

    print("[metadata] downloading cohort metadata")
    ok = metadata.download_metadata()
    print("  done" if ok else "  completed with errors")


def stage_annotations(args):
    from pipeline import metadata

    print("[annotations] syncing technician annotation CSVs")
    ok = metadata.download_annotations()
    print("  done" if ok else "  completed with errors")


def stage_index(args):
    from pipeline import index_annotations

    print("[index] parsing annotation CSVs -> labels")
    labelled = index_annotations.build_index()
    print(index_annotations.summarize(labelled))


def stage_select(args):
    from pipeline import selection

    print("[select] choosing patient-level subset")
    selection.select_patients(cfg=_selection_cfg(args))


def stage_split(args):
    from pipeline import selection

    print("[split] assigning leakage-safe train/val/test")
    selection.make_splits(cfg=_selection_cfg(args))


def stage_download(args):
    from pipeline import download

    print("[download] fetching selected EDF recordings")
    splits = tuple(args.splits) if getattr(args, "splits", None) else None
    download.download_selection(
        dl=_download_cfg(args), splits=splits, dry_run=args.dry_run
    )


def stage_fullscale(args):
    from pipeline import fullscale, selection

    if not getattr(args, "skip_select", False):
        print("[fullscale] selecting all annotated patients")
        selection.select_all_patients(annotated_only=True)
        selection.make_splits(cfg=_selection_cfg(args))
    print("[fullscale] starting rolling ingestion")
    fullscale.run_fullscale(
        skip_phase_a=args.skip_phase_a,
        batch_gb=args.batch_gb,
        header_workers=args.header_workers,
        cache_workers=args.cache_workers,
        download_config=_download_cfg(args),
    )


def stage_all(args):
    stage_manifest(args)
    if not args.offline:
        stage_metadata(args)
        stage_annotations(args)
    stage_index(args)
    stage_select(args)
    stage_split(args)
    if not args.skip_download and not args.offline:
        stage_download(args)
    else:
        print("[all] skipping download stage "
              "(run `download` when ready to fetch signals).")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Neurotech EEG 100 GB acquisition pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="stage", required=True)

    def add_select_opts(sp):
        sp.add_argument("--target-gb", type=float, help="Download budget (GB).")
        sp.add_argument("--positive-fraction", type=float,
                        help="Target seizure byte fraction (0-1).")
        sp.add_argument("--seed", type=int, help="RNG seed.")

    def add_download_opts(sp):
        sp.add_argument("--workers", type=int, help="Parallel download workers.")
        sp.add_argument("--splits", nargs="+",
                        choices=["train", "val", "test"],
                        help="Restrict download to these splits.")
        sp.add_argument("--dry-run", action="store_true",
                        help="Report what would download, fetch nothing.")

    sub.add_parser("manifest").set_defaults(func=stage_manifest)
    sub.add_parser("metadata").set_defaults(func=stage_metadata)
    sub.add_parser("annotations").set_defaults(func=stage_annotations)
    sub.add_parser("index").set_defaults(func=stage_index)

    sp_sel = sub.add_parser("select")
    add_select_opts(sp_sel)
    sp_sel.set_defaults(func=stage_select)

    sp_split = sub.add_parser("split")
    add_select_opts(sp_split)
    sp_split.set_defaults(func=stage_split)

    sp_dl = sub.add_parser("download")
    add_download_opts(sp_dl)
    sp_dl.set_defaults(func=stage_download)

    sp_all = sub.add_parser("all")
    add_select_opts(sp_all)
    add_download_opts(sp_all)
    sp_all.add_argument("--skip-download", action="store_true",
                        help="Run planning stages but not the big download.")
    sp_all.add_argument("--offline", action="store_true",
                        help="Only run stages that need no network "
                             "(manifest/index/select/split).")
    sp_all.set_defaults(func=stage_all)

    sp_fs = sub.add_parser(
        "fullscale",
        help="Full dataset ingestion: header manifest + rolling cache (EC2-scale)",
    )
    add_select_opts(sp_fs)
    sp_fs.add_argument("--batch-gb", type=float, default=300.0,
                       help="Rolling batch size in GB (default 300).")
    sp_fs.add_argument("--header-workers", type=int, default=32,
                       help="Parallel workers for header stub downloads.")
    sp_fs.add_argument("--cache-workers", type=int,
                       help="Parallel workers for cache fill (default: CPU-2).")
    sp_fs.add_argument("--workers", type=int, help="Parallel download workers.")
    sp_fs.add_argument("--skip-phase-a", action="store_true",
                       help="Reuse existing windows.parquet + allocated cache.")
    sp_fs.add_argument("--skip-select", action="store_true",
                       help="Reuse existing splits.parquet.")
    sp_fs.set_defaults(func=stage_fullscale)

    return p


def main(argv=None) -> int:
    config.ensure_dirs()
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
