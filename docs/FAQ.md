# FAQ

Common design questions for this project. This repository is Kush Patel's ML pipeline on the public Neurotech EEG corpus (BDSP). It is not the dataset authors' BIDS conversion tooling ([`bdsp-core/Neurotech-EEG-Wrangling`](https://github.com/bdsp-core/Neurotech-EEG-Wrangling)).

### What does the model actually detect?
Windows near technician workflow seizure markers (+/- 30 s). That is seizure-marker-proximity detection, not multi-expert clinically validated seizure-interval detection.

### Why patient-level splitting?
The same patient can contribute many recordings. Splitting by recording alone can leak patient-specific artifacts into the test set. Patient-level splits measure generalization to unseen people.

### Why focal loss?
Positives are rare (~2-4% of windows). Focal loss down-weights easy negatives so training focuses on harder examples, alongside balanced sampling and a positive class weight.

### Why PR-AUC (and PRG) instead of only ROC?
With ~96% negatives, ROC can look strong while precision is poor. PR-AUC measures how well marker-proximal windows rank at the top. PRG-AUC is included because validation and test prevalences differ.

### Why 10-second windows?
Short enough to localize events; long enough to capture evolving morphology on scalp EEG. A 5 s stride creates overlap so inference can temporally smooth. Similar lengths (~8-12 s) are common in scalp detection work.

### Why +/- 30 seconds around markers?
Xltek annotations are point markers, not onset/offset intervals. A +/- 30 s radius is a weak-supervision prior ("near a technician seizure mark"). Wider radii add label noise; narrower radii starve positives.

### How does temporal smoothing help?
Overlapping windows are scored independently. Pooling over +/- 15 s suppresses isolated spikes and favors sustained elevations. On the held-out test set, Conformer PR-AUC rose from ~0.32 (raw) to ~0.46 (15 s max-smooth).

### Why a ~244 GB window cache?
Re-opening, filtering, and resampling EDFs every epoch is too slow at this scale. A one-time preprocess to memmap float arrays makes training a sequential read. Raw EDFs were deleted after cache fill; the cache can be archived separately (not in git).

### How is leakage controlled?
1. Patient IDs never cross splits
2. Cache coverage checks against the window table
3. Thresholds chosen on validation, reported once on test

### What are the classical baselines?

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| Logistic regression on bandpower | 0.082 | 0.680 |
| EEGNet-style CNN | 0.174 | 0.757 |
| EEG Conformer (raw) | 0.318 | 0.780 |
| EEG Conformer + 15 s max-smooth | 0.460 | 0.859 |

The Conformer + smoothing stack improves 0.38 PR-AUC over bandpower logistic regression and 0.29 over EEGNet on the same held-out patients.

### Why was forecasting near chance?
Scalp EEG, weak point markers, and sparse preictal sampling did not yield a usable patient-independent preictal signal at multi-minute horizons. Handcrafted spectral features also failed near chance, so the result is not explained by architecture alone. Approximate onsets and limited ambulatory context likely contribute. The null is reported intentionally: detection is not forecasting here.

### What are reasonable next steps?
Expert-adjudicated intervals on a subset; denser preictal windows; patient-specific calibration; FA-budgeted operating points; comparison on TUH / CHB-MIT under the same metric suite.
