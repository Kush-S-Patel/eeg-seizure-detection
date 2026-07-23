# Improvement Roadmap

The starter deliberately separates stable interfaces from research choices.
Prioritize label quality and external validation before increasing model size.

## 1. Label quality

- Replace point-marker expansion with neurologist-reviewed onset/duration
  intervals.
- Record inter-rater agreement and retain uncertain/disputed labels.
- Distinguish electrographic seizures from patient events and clinical notes.
- Add hard-negative review for rhythmic artifacts and benign variants.

Extension point: implement `LabelPolicy` in
`src/seizure_detector/interfaces.py`.

## 2. Event detection

- Merge consecutive positive windows into candidate events.
- Tune hysteresis and minimum-duration rules on validation patients only.
- Report event sensitivity, onset latency, and false alarms per 24 hours.
- Add streaming state so overlapping windows do not generate duplicate alarms.

## 3. Multi-task EEG interpretation

- Add spike, sharp-wave, sleep stage, and artifact heads.
- Use shared representations with task-specific loss masks.
- Never treat an absent workflow annotation as a confirmed negative.

Extension point: register models in `src/seizure_detector/models/__init__.py`.

## 4. Representation learning

- Self-supervised pretraining on unlabeled windows.
- ~~Channel dropout and montage-aware augmentation~~ — done:
  `--augment-channel-dropout` / `--augment-noise-std` in `TrainConfig`.
- ~~Compare temporal CNNs, channel attention, and efficient Transformers~~ —
  started: `eegnet` (compact), `eegnet_deep` (wider/deeper), and `cnn_gru`
  (temporal CNN + attention-pooled BiGRU) are registered and swappable via
  `--model`; `eegnet_deep` currently wins on val PR-AUC/F1 (see README). A
  channel-attention or small Transformer encoder is a natural next entry —
  register it in `models/__init__.py` with the same `forward(x, channel_mask)`
  contract and it gets checkpointing/scheduling/early-stopping for free.
- Track patient-level confidence intervals, not only point estimates.
- Revisit `--loss focal` / `--auto-pos-weight` once labels are stronger
  (Section 1); on today's point-marker labels neither beat the balanced
  sampler alone.

## 5. Signal quality and robustness

- Learn or validate artifact rejection for electrode pops, flat channels, EMG,
  movement, and mains noise.
- Carry the channel-presence mask into richer model architectures.
- Calibrate performance by recording type and duration.
- Test robustness to missing electrodes and alternate reference schemes.

Extension point: implement `Preprocessor` and quality hooks in
`src/seizure_detector/interfaces.py`.

## 6. Validation and operations

- Validate on a separate institution/device before any clinical claim.
- Add experiment tracking, immutable data/model versions, and model cards.
- Add probability calibration and drift monitoring.
- Add live-device ingestion only after offline event logic is validated.
- Obtain regulatory, privacy, cybersecurity, and human-factors review before
  considering clinical deployment.
