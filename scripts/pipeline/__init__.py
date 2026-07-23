"""Neurotech EEG data-acquisition pipeline.

A staged, resumable pipeline that turns the raw S3 listing (``files.txt``)
into a leakage-safe, class-balanced ~100 GB subset for training a seizure
detector.

Stages (see ``scripts/downloader_100gb.py`` for the CLI orchestrator):

    manifest    Parse files.txt -> per-recording manifest.
    metadata    Download small cohort metadata (participants, phenotype, ...).
    annotations Sync every technician annotation CSV (tiny) for labelling.
    index       Parse annotation CSVs -> per-recording seizure/spike labels.
    select      Patient-level sampling + positive oversampling under a GB budget.
    split       Patient-level train/val/test split (no data leakage).
    download    Download the selected EDF recordings (parallel, resumable).
"""

from . import config

__all__ = ["config"]
