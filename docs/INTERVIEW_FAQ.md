# Interview notes — common questions

Short answers you can use in interviews. This project is **Kush Patel's** applied ML / data-engineering portfolio work on the public Neurotech EEG corpus (BDSP). It is **not** the dataset authors' BIDS conversion tooling ([`bdsp-core/Neurotech-EEG-Wrangling`](https://github.com/bdsp-core/Neurotech-EEG-Wrangling)).

## Framing first (say this out loud)

> I built a **seizure-marker-proximity detector**: the model scores 10‑second windows by how seizure-like they look relative to technician workflow markers (±30 s). That is **not** the same as clinically validated seizure-interval detection with multi-expert labels.

---

### Why patient-level splitting?
Same patient can contribute many recordings. If one patient's sessions land in both train and test, the model can memorize patient-specific artifacts (impedance, montage quirks, background rhythm) and look great for the wrong reason. Patient-level splits force generalization to **unseen people**.

### Why focal loss?
Positives are rare (~2–4% of windows). Plain BCE is dominated by easy negatives. Focal loss down-weights those easy examples so gradients focus on hard / rare positives — complementary to balanced sampling and `pos_weight`.

### Why PR-AUC instead of (only) ROC?
With ~96% negatives, ROC can look strong while precision is terrible. PR-AUC asks whether true marker-proximal windows actually rise to the top of the ranked list. We also report **PRG-AUC** because val/test prevalences differ.

### Why 10-second windows?
Short enough to localize events; long enough to capture evolving seizure morphology on scalp EEG. 5 s stride gives overlap so inference can temporally smooth. Literature often uses ~8–12 s for scalp detection.

### Why ±30 seconds around markers?
Xltek annotations are **point** markers, not onset/offset intervals. A ±30 s radius is a weak-supervision prior: “near a technician seizure mark.” Wider radii add label noise; narrower radii starve positives.

### How did smoothing improve results?
Overlapping windows are scored independently. Max/mean pooling over ±15 s suppresses isolated blips and boosts sustained elevations — closer to how events are read clinically. On test, Conformer PR-AUC rose from **~0.32 (raw)** to **~0.46 (15 s max-smooth)**.

### Why cache ~244 GB instead of preprocessing online?
Naive dataloaders re-opened EDFs, filtered, and resampled every epoch — prohibitively slow at this scale. One-time preprocess → memmap floats turns epochs into sequential array reads. Raw EDFs were deleted after cache fill; the cache was archived to S3.

### How did you prevent leakage?
1. Patient IDs never cross splits  
2. Fingerprinted window tables / cache coverage checks  
3. Thresholds chosen on **val**, reported once on **test**  
4. No test peeking for architecture selection beyond that discipline  

### What are the classical baselines?
| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| Logistic regression on bandpower | **0.082** | 0.680 |
| EEGNet-style CNN | 0.174 | 0.757 |
| EEG Conformer (raw) | 0.318 | 0.780 |
| EEG Conformer + 15 s max-smooth | **0.460** | **0.859** |

**Soundbite:** Conformer + smoothing beats spectral logistic regression by **+0.38 PR-AUC** and beats EEGNet by **+0.29 PR-AUC** on the same patient-held-out test set — so the extra complexity is earned, not cosmetic.

### Why was forecasting near chance?
Short version: **scalp EEG + weak point markers + sparse preictal sampling** did not yield a usable patient-independent preictal signature at 3–30 minute horizons. Handcrafted spectral features also failed (~chance), so this was not just “wrong architecture.” Horizon/label noise and limited ambulatory context likely dominate. Reporting the null is intentional.

### What would you do next with more time?
Expert-adjudicated intervals on a subset; denser preictal windows; patient-specific calibration; prospective FA-budget constraints; compare against TUH/CHB-MIT under the same metric suite.
