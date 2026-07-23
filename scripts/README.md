# Neurotech EEG — 100 GB Acquisition Pipeline

Builds a **leakage-safe, class-balanced ~100 GB subset** of the
[Neurotech EEG Dataset](https://doi.org/10.60508/v99k-ek82) (10.2 TB, 23,607
recordings, 4,914 patients) for training a seizure detector.

The old `downloader_100gb.py` downloaded EDFs in *file order* until it hit
100 GB. That is unsafe for ML:

- **Data leakage** — consecutive files belong to the same patient, so the same
  patient ends up in both train and test.
- **Class imbalance** — only ~29% of recordings contain seizures; a naive
  prefix scan can grab almost none.

This pipeline replaces that with staged, resumable acquisition.

## Requirements

- Python deps: `pip install -r requirements.txt`
- [AWS CLI v2](https://docs.aws.amazon.com/cli/) with credentials + an accepted
  BDSP data-use agreement (required only for the download stages).
- `data/files.txt` — the recursive S3 listing:

```bash
aws s3 ls s3://bdsp-credentialed-ac-.../EEG/bids/Neurotech/ --recursive > files.txt
```

## Stages

| Stage | Network? | What it does |
|-------|----------|--------------|
| `manifest`    | no  | Parse `files.txt` → one row per recording (patient, session, size, sidecars). Filters header-only stub EDFs. |
| `metadata`    | yes | Download cohort metadata (`participants`, `phenotype/`, `README`). A few MB. |
| `annotations` | yes | Sync **every** technician annotation CSV (`*_Xltek.csv`, ~14.5k tiny files) in one pass. |
| `index`       | no  | Parse annotation CSVs → per-recording seizure/spike/sharp labels. |
| `select`      | no  | **Patient-level** sampling with seizure-positive oversampling under the GB budget. |
| `split`       | no  | **Patient-level** train/val/test split (stratified, asserts zero leakage). |
| `download`    | yes | Download the selected EDFs + sidecars (parallel, resumable). |
| `all`         | —   | Run everything end to end. |

## Usage

```bash
# 1. Plan everything except the big download (needs annotations synced first
#    for real labels; runs offline otherwise).
python scripts/downloader_100gb.py all --skip-download

# 2. Inspect data/artifacts/ (manifest, annotation_index, selected_recordings,
#    splits), then fetch signals — start with just the training split:
python scripts/downloader_100gb.py download --splits train
python scripts/downloader_100gb.py download          # remaining splits

# Tuning
python scripts/downloader_100gb.py select --target-gb 100 --positive-fraction 0.5 --seed 1337
python scripts/downloader_100gb.py download --workers 12 --dry-run
```

Every stage writes an artifact to `data/artifacts/` (Parquet, or CSV if
`pyarrow` is absent) and is idempotent — re-running skips work already done.

## How leakage & imbalance are handled

- **No leakage:** the atomic unit is a *patient*. When a patient is selected we
  take all their sessions, and train/val/test are assigned per patient. A
  runtime assertion fails the split if any patient appears in two sets.
- **Class balance:** the byte budget is split between seizure-positive patients
  and seizure-free patients (default 50/50), heavily oversampling the ~29%
  positive base rate. Because we sample whole patients, the seizure-*recording*
  fraction can't be forced to exactly 50%; final window-level balancing is done
  during training (e.g. weighted sampling of seizure vs. non-seizure epochs).

## Notes / assumptions

- Header-only stub EDFs are excluded via a size floor (`MIN_SIGNAL_BYTES` in
  `pipeline/config.py`); `s3 ls` doesn't expose `n_records`.
- The Xltek CSV schema isn't published, so labelling scans cells as free text
  against keyword sets (with a negation guard). Refine
  `SEIZURE_KEYWORDS`/`NEGATION_PREFIXES` in `config.py` once a few real CSVs are
  inspected.
- Set `NEUROTECH_ACCESS_POINT` / `AWS_CLI_PATH` env vars to override defaults.
