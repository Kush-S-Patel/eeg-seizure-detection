"""Phase 1 - download the small stuff first.

Two cheap stages that must run before any EDF is fetched:

* ``download_metadata``   cohort-level files (participants, phenotype/, README).
* ``download_annotations`` every technician annotation CSV (~14.5k tiny files),
  synced in a single listing pass so we can label recordings *before*
  committing 100 GB of signal downloads.
"""

from __future__ import annotations

from . import awscli, config


def download_metadata() -> bool:
    """Fetch participants/phenotype/README/dataset_description (a few MB)."""
    ok = True
    for name in config.METADATA_FILES:
        key = f"{config.S3_PREFIX}/{name}"
        local = config.RAW_DIR / key
        print(f"  -> {name}")
        ok &= awscli.cp(key, local, max_retries=config.DEFAULT_DOWNLOAD.max_retries)

    # phenotype/ is a handful of small tsv/json files -> sync wholesale.
    print(f"  -> {config.PHENOTYPE_DIR}/ (all tsv/json)")
    pheno_prefix = f"{config.S3_PREFIX}/{config.PHENOTYPE_DIR}"
    ok &= awscli.sync(pheno_prefix, config.RAW_DIR / pheno_prefix)
    return ok


def download_annotations() -> bool:
    """Sync every ``*_Xltek.csv`` under the dataset in one pass.

    These are 1-2 KB each; the whole set is only tens of MB but lets us index
    seizure/spike/sharp labels before selecting EDFs.
    """
    print("  syncing all *_Xltek.csv (single recursive pass)...")
    return awscli.sync(
        config.S3_PREFIX,
        config.RAW_DIR / config.S3_PREFIX,
        include="*_Xltek.csv",
        exclude_all_first=True,
    )
