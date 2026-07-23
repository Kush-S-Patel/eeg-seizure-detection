"""Typed configuration for preparation, preprocessing, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
OUTPUT_DIR = ROOT / "outputs"

SPLITS_PATH = ARTIFACTS_DIR / "splits.parquet"
WINDOWS_PATH = ARTIFACTS_DIR / "windows.parquet"
WINDOW_CACHE_DIR = ARTIFACTS_DIR / "window_cache"


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: float = 10.0
    stride_seconds: float = 5.0
    positive_radius_seconds: float = 30.0
    negative_guard_seconds: float = 300.0
    max_negative_windows_per_recording: int = 200
    seed: int = 1337


@dataclass(frozen=True)
class SignalConfig:
    sample_rate: int = 128
    low_hz: float = 0.5
    high_hz: float = 45.0
    notch_hz: float = 60.0
    clip: float = 8.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    num_workers: int = 0
    seed: int = 1337
    max_train_windows: int | None = None
    device: str = "auto"

    # --- Model ---
    # See `seizure_detector.models.MODELS` for the registry; `register_model`
    # lets you add architectures without touching training code. Defaults to
    # the deeper variant: on this dataset it beat the compact "eegnet" and
    # "cnn_gru" on held-out val PR-AUC/F1 with a plateau LR schedule (see
    # docs/ROADMAP.md). "eegnet" is still a good choice for quick iteration.
    model_name: str = "eegnet_deep"
    dropout: float = 0.35

    # --- Class imbalance ---
    # These stack with each other: the sampler changes how often a positive
    # window is *seen*, pos_weight changes how much its loss *counts*. Using
    # both aggressively can overcorrect, so auto_pos_weight defaults off.
    use_balanced_sampler: bool = True
    pos_weight: float | None = None
    auto_pos_weight: bool = False

    # --- Loss / regularization ---
    loss_fn: str = "bce"  # "bce" | "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = 1.0
    augment_noise_std: float = 0.0
    augment_channel_dropout: float = 0.0
    augment_time_flip: float = 0.0
    mixup_alpha: float = 0.0

    # --- LR schedule ---
    lr_scheduler: str = "plateau"  # "none" | "plateau" | "cosine" | "onecycle"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # --- Early stopping ---
    early_stopping_metric: str = "pr_auc"  # "pr_auc" | "roc_auc" | "f1" | "loss"
    early_stopping_min_delta: float = 0.0

    # --- Stability / resume ---
    ema_decay: float = 0.0
    resume_from: str | None = None
    record_norm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
