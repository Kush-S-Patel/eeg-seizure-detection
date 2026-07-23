"""Training, checkpointing, evaluation, and prediction."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cache import ensure_window_cache
from .config import OUTPUT_DIR, SignalConfig, TrainConfig, WINDOW_CACHE_DIR
from .dataset import EEGWindowDataset, balanced_sampler, limit_windows
from .losses import build_loss_fn, resolve_pos_weight, smooth_targets
from .metrics import (
    assign_event_ids,
    binary_metrics,
    bootstrap_metrics,
    choose_threshold,
    event_metrics_from_windows,
)
from .models import create_model

_HIGHER_IS_BETTER = {"pr_auc", "roc_auc", "f1"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(
    table: pd.DataFrame,
    signal: SignalConfig,
    train: bool,
    cfg: TrainConfig,
    cache_dir: Path = WINDOW_CACHE_DIR,
) -> DataLoader:
    dataset = EEGWindowDataset(
        table, signal, cache_dir=cache_dir, record_norm=cfg.record_norm
    )
    sampler = balanced_sampler(table["label"], cfg.seed) if train and cfg.use_balanced_sampler else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=train and sampler is None,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    if cfg.lr_scheduler == "none":
        return None
    if cfg.lr_scheduler == "plateau":
        mode = "min" if cfg.early_stopping_metric == "loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience, min_lr=cfg.min_lr,
        )
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr
        )
    if cfg.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=max(cfg.epochs * max(steps_per_epoch, 1), 1),
        )
    raise ValueError(f"Unknown lr_scheduler {cfg.lr_scheduler!r}")


def _augment(x: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.augment_noise_std > 0:
        x = x + torch.randn_like(x) * cfg.augment_noise_std
    if cfg.augment_channel_dropout > 0:
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= cfg.augment_channel_dropout)
        x = x * keep.to(x.dtype)
    if cfg.augment_time_flip > 0:
        flip = torch.rand(x.shape[0], 1, 1, device=x.device) < cfg.augment_time_flip
        x = torch.where(flip, x.flip(-1), x)
    return x


def _mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed inputs and the two label tensors with mix coefficient."""
    if alpha <= 0 or x.shape[0] < 2:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[index], y, y[index], lam


class ModelEMA:
    """Exponential moving average of model weights (eval-time snapshot)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict({**model.state_dict(), **backup}, strict=False)


def _score(metric: str, metrics: dict[str, float] | None, loss: float) -> float:
    """Uniformly higher-is-better score used for checkpointing/early stopping."""
    if metric == "loss" or metrics is None:
        return -loss
    return metrics.get(metric, -loss)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    loss_fn: nn.Module,
    *,
    train_cfg: TrainConfig | None = None,
    batch_scheduler=None,
    ema: ModelEMA | None = None,
    tta_flip: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["channel_mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mix_y_b = None
        lam = 1.0
        if training and train_cfg is not None:
            x = _augment(x, train_cfg)
            if train_cfg.mixup_alpha > 0:
                x, y, mix_y_b, lam = _mixup(x, y, train_cfg.mixup_alpha)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x, mask)
                if not training and tta_flip:
                    logits = 0.5 * (logits + model(x.flip(-1), mask))
                loss_targets = (
                    smooth_targets(y, train_cfg.label_smoothing)
                    if training and train_cfg is not None
                    else y
                )
                if mix_y_b is not None and training and train_cfg is not None:
                    loss_b = smooth_targets(mix_y_b, train_cfg.label_smoothing)
                    loss = lam * loss_fn(logits, loss_targets) + (1.0 - lam) * loss_fn(logits, loss_b)
                else:
                    loss = loss_fn(logits, loss_targets)
            if training:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                if train_cfg is not None and train_cfg.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                if batch_scheduler is not None:
                    batch_scheduler.step()
        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        probabilities.append(np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0))
        targets.append(y.detach().cpu().numpy())
        indices.append(batch["index"].numpy())
    return (
        float(np.nanmean(losses)) if losses else float("nan"),
        np.concatenate(probabilities) if probabilities else np.empty(0),
        np.concatenate(targets) if targets else np.empty(0),
        np.concatenate(indices) if indices else np.empty(0, dtype=int),
    )


def train_model(
    windows: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR / "baseline",
    *,
    train_config: TrainConfig = TrainConfig(),
    signal_config: SignalConfig = SignalConfig(),
    model_name: str | None = None,
    cache_dir: Path = WINDOW_CACHE_DIR,
    rebuild_cache: bool = False,
) -> Path:
    seed_everything(train_config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or train_config.model_name

    # One-time pass: filter/resample each unique EDF once and cache every
    # window as a plain array. Without this, every epoch re-reads and
    # re-filters every window from scratch, which is what made training slow.
    cache_workers = train_config.num_workers if train_config.num_workers > 0 else max(
        1, (__import__("os").cpu_count() or 4) - 2
    )
    ensure_window_cache(
        windows,
        signal_config,
        cache_dir,
        rebuild=rebuild_cache,
        workers=cache_workers,
    )

    train_table = limit_windows(
        windows[windows["split"] == "train"], train_config.max_train_windows, train_config.seed
    )
    val_table = windows[windows["split"] == "val"].reset_index(drop=True)
    if train_table.empty:
        raise ValueError("No training windows available")

    device = resolve_device(train_config.device)
    model = create_model(
        model_name,
        channels=18,
        samples=int(signal_config.sample_rate * float(train_table.duration_seconds.iloc[0])),
        dropout=train_config.dropout,
    ).to(device)
    if train_config.resume_from:
        resume_path = Path(train_config.resume_from)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
        print(
            f"Resumed weights from {resume_path} "
            f"(missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=train_config.ema_decay) if train_config.ema_decay > 0 else None
    train_loader = _loader(train_table, signal_config, True, train_config, cache_dir)
    val_loader = (
        _loader(val_table, signal_config, False, train_config, cache_dir) if len(val_table) else None
    )
    scheduler = _build_scheduler(optimizer, train_config, steps_per_epoch=len(train_loader))
    batch_scheduler = scheduler if train_config.lr_scheduler == "onecycle" else None

    pos_weight = resolve_pos_weight(
        train_table["label"],
        pos_weight=train_config.pos_weight,
        auto_pos_weight=train_config.auto_pos_weight,
    )
    loss_fn = build_loss_fn(
        train_config.loss_fn, pos_weight=pos_weight, focal_gamma=train_config.focal_gamma, device=device
    )
    if pos_weight is not None:
        print(f"Using pos_weight={pos_weight:.2f} with loss_fn={train_config.loss_fn!r}", flush=True)
    if ema is not None:
        print(f"Using EMA decay={train_config.ema_decay}", flush=True)

    history: list[dict] = []
    best_score = -float("inf")
    stale = 0
    checkpoint = output_dir / "best.pt"
    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_prob, train_y, _ = _epoch(
            model, train_loader, device, optimizer, scaler, loss_fn,
            train_cfg=train_config, batch_scheduler=batch_scheduler, ema=ema,
        )
        row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        threshold = 0.5
        val_metrics: dict[str, float] | None = None
        backup = None
        # Skip cold EMA for the first epoch (shadow still near init when training from scratch).
        use_ema_eval = ema is not None and epoch >= 2
        if use_ema_eval:
            backup = ema.copy_to(model)
        if val_loader is not None:
            val_loss, val_prob, val_y, _ = _epoch(
                model, val_loader, device, None, scaler, loss_fn, tta_flip=True
            )
            threshold = choose_threshold(val_y, val_prob)
            val_metrics = binary_metrics(val_y, val_prob, threshold)
            row.update({"val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        else:
            val_loss = train_loss
            train_metrics = binary_metrics(train_y, train_prob, threshold)
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
        score = _score(train_config.early_stopping_metric, val_metrics, val_loss)
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"score={score:.4f} threshold={threshold:.3f} lr={row['lr']:.2e}",
            flush=True,
        )
        if scheduler is not None:
            if train_config.lr_scheduler == "plateau":
                scheduler.step(score if train_config.early_stopping_metric != "loss" else val_loss)
            elif train_config.lr_scheduler == "cosine":
                scheduler.step()
        if np.isfinite(score) and score > best_score + train_config.early_stopping_min_delta:
            best_score = score
            stale = 0
            if use_ema_eval and ema is not None:
                save_state = {k: v.detach().clone() for k, v in ema.shadow.items()}
            else:
                save_state = model.state_dict()
            torch.save(
                {
                    "model_state": save_state,
                    "model_name": model_name,
                    "threshold": threshold,
                    "signal_config": asdict(signal_config),
                    "train_config": train_config.to_dict(),
                    "window_seconds": float(train_table.duration_seconds.iloc[0]),
                    "used_ema": use_ema_eval,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= train_config.patience:
                print("Early stopping", flush=True)
                if backup is not None:
                    ModelEMA.restore(model, backup)
                break
        if backup is not None:
            ModelEMA.restore(model, backup)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {"train": train_config.to_dict(), "signal": asdict(signal_config), "model": model_name},
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint


def load_checkpoint(path: Path, device: torch.device | None = None):
    device = device or resolve_device("auto")
    payload = torch.load(path, map_location=device, weights_only=False)
    signal = SignalConfig(**payload["signal_config"])
    dropout = payload.get("train_config", {}).get("dropout", 0.35)
    model = create_model(
        payload["model_name"],
        channels=18,
        samples=int(signal.sample_rate * payload["window_seconds"]),
        dropout=dropout,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, signal, float(payload.get("threshold", 0.5)), device


def smooth_probabilities_by_recording(
    predictions: pd.DataFrame,
    *,
    smooth_seconds: float,
    mode: str = "softmax",
    temperature: float = 0.15,
    rethreshold: bool = True,
) -> pd.DataFrame:
    """Pool overlapping-window probabilities within each recording.

    ``mode``:
      - ``mean``: arithmetic mean (can dilute sparse positives)
      - ``max``: keep the strongest nearby score
      - ``softmax``: temperature-weighted soft-max (default; better for PR-AUC)
    """
    if smooth_seconds <= 0 or predictions.empty:
        return predictions
    if "record_id" not in predictions.columns or "start_seconds" not in predictions.columns:
        return predictions

    output = predictions.copy().reset_index(drop=True)
    radius = smooth_seconds / 2.0
    smoothed = np.empty(len(output), dtype=np.float64)
    for _, group in output.groupby("record_id", sort=False):
        starts = group["start_seconds"].to_numpy(dtype=np.float64)
        probs = group["probability"].to_numpy(dtype=np.float64)
        idx = group.index.to_numpy()
        order = np.argsort(starts)
        starts_s = starts[order]
        probs_s = probs[order]
        idx_s = idx[order]
        for i, t in enumerate(starts_s):
            left = np.searchsorted(starts_s, t - radius, side="left")
            right = np.searchsorted(starts_s, t + radius, side="right")
            window = probs_s[left:right]
            if mode == "max":
                smoothed[idx_s[i]] = float(window.max())
            elif mode == "mean":
                smoothed[idx_s[i]] = float(window.mean())
            else:
                # Softmax pooling over neighbor probs (stable for ranking).
                scaled = (window - window.max()) / max(temperature, 1e-6)
                weights = np.exp(scaled)
                smoothed[idx_s[i]] = float((weights * window).sum() / weights.sum())
    output["probability_raw"] = output["probability"]
    output["probability"] = smoothed
    if rethreshold and "label" in output.columns and len(np.unique(output["label"])) > 1:
        threshold = choose_threshold(output["label"].to_numpy(), smoothed)
        output["threshold"] = threshold
    else:
        threshold = float(output["threshold"].iloc[0])
    output["prediction"] = (output["probability"] >= threshold).astype(int)
    return output


def predict_windows(
    checkpoint: Path,
    table: pd.DataFrame,
    *,
    batch_size: int = 64,
    cache_dir: Path = WINDOW_CACHE_DIR,
    smooth_seconds: float = 0.0,
    smooth_mode: str = "softmax",
    tta_flip: bool = True,
) -> pd.DataFrame:
    model, signal, threshold, device = load_checkpoint(checkpoint)
    cfg = TrainConfig(batch_size=batch_size, device=str(device))
    loader = _loader(table.reset_index(drop=True), signal, False, cfg, cache_dir)
    loss_fn = nn.BCEWithLogitsLoss()
    _, probabilities, targets, indices = _epoch(
        model, loader, device, None, None, loss_fn, tta_flip=tta_flip
    )
    output = table.reset_index(drop=True).iloc[indices].copy()
    output["probability"] = probabilities
    output["prediction"] = (probabilities >= threshold).astype(int)
    output["threshold"] = threshold
    output["label"] = targets.astype(int)
    return smooth_probabilities_by_recording(
        output, smooth_seconds=smooth_seconds, mode=smooth_mode
    )


def _enrich_metrics(predictions: pd.DataFrame, metrics: dict, *, n_boot: int = 0) -> dict:
    """Add SzCORE-style event metrics and optional event-level bootstrap CIs."""
    duration_hours = float(predictions["duration_seconds"].sum() / 3600)
    thr = float(predictions["threshold"].iloc[0])
    y = predictions["label"].to_numpy()
    p = predictions["probability"].to_numpy()
    event = event_metrics_from_windows(
        predictions["record_id"].to_numpy(),
        predictions["start_seconds"].to_numpy(),
        predictions["duration_seconds"].to_numpy(),
        y,
        p,
        thr,
        duration_hours,
    )
    metrics.update(event)
    if n_boot > 0:
        eids = assign_event_ids(
            predictions["record_id"].to_numpy(),
            predictions["start_seconds"].to_numpy(),
            y,
        )
        boot = bootstrap_metrics(
            y,
            p,
            event_ids=eids,
            record_ids=predictions["record_id"].to_numpy(),
            duration_hours=duration_hours,
            n_boot=n_boot,
            threshold=thr,
        )
        for key, stats in boot.items():
            metrics[f"boot_{key}_mean"] = stats["mean"]
            metrics[f"boot_{key}_lo"] = stats["lo"]
            metrics[f"boot_{key}_hi"] = stats["hi"]
    return metrics


def evaluate_checkpoint(
    checkpoint: Path,
    windows: pd.DataFrame,
    split: str,
    output_dir: Path,
    *,
    smooth_seconds: float = 0.0,
    smooth_mode: str = "softmax",
    tta_flip: bool = True,
    n_boot: int = 0,
) -> dict[str, float]:
    table = windows[windows["split"] == split].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"No local {split} windows; download/audit that split first")
    predictions = predict_windows(
        checkpoint,
        table,
        smooth_seconds=smooth_seconds,
        smooth_mode=smooth_mode,
        tta_flip=tta_flip,
    )
    duration_hours = predictions["duration_seconds"].sum() / 3600
    metrics = binary_metrics(
        predictions["label"].to_numpy(),
        predictions["probability"].to_numpy(),
        float(predictions["threshold"].iloc[0]),
        duration_hours,
    )
    metrics = _enrich_metrics(predictions, metrics, n_boot=n_boot)
    if smooth_seconds > 0:
        metrics["smooth_seconds"] = float(smooth_seconds)
        metrics["smooth_mode"] = smooth_mode
    metrics["tta_flip"] = bool(tta_flip)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_smooth{int(smooth_seconds)}s" if smooth_seconds > 0 else ""
    predictions.to_csv(output_dir / f"{split}_predictions{suffix}.csv", index=False)
    (output_dir / f"{split}_metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def evaluate_ensemble(
    checkpoints: list[Path],
    windows: pd.DataFrame,
    split: str,
    output_dir: Path,
    *,
    weights: list[float] | None = None,
    smooth_seconds: float = 0.0,
    smooth_mode: str = "softmax",
    tta_flip: bool = True,
    n_boot: int = 0,
) -> dict[str, float]:
    """Blend probabilities from multiple checkpoints (val-tuned weights recommended)."""
    table = windows[windows["split"] == split].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"No local {split} windows; download/audit that split first")
    if not checkpoints:
        raise ValueError("Need at least one checkpoint for ensemble evaluation")
    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)
    if len(weights) != len(checkpoints):
        raise ValueError("weights length must match checkpoints")
    weight_sum = float(sum(weights))
    weights = [w / weight_sum for w in weights]

    blended = None
    labels = None
    meta = None
    for ckpt, weight in zip(checkpoints, weights):
        pred = predict_windows(
            ckpt, table, smooth_seconds=0.0, tta_flip=tta_flip
        )
        probs = pred["probability"].to_numpy(dtype=np.float64) * weight
        if blended is None:
            blended = probs
            labels = pred["label"].to_numpy(dtype=int)
            meta = pred.drop(columns=["probability", "prediction", "threshold"], errors="ignore")
        else:
            blended = blended + probs

    assert blended is not None and labels is not None and meta is not None
    output = meta.copy()
    output["probability"] = blended
    output["label"] = labels
    output["threshold"] = 0.5
    if smooth_seconds > 0:
        output = smooth_probabilities_by_recording(
            output, smooth_seconds=smooth_seconds, mode=smooth_mode, rethreshold=True
        )
    elif len(np.unique(labels)) > 1:
        thr = choose_threshold(labels, blended)
        output["threshold"] = thr
        output["prediction"] = (output["probability"] >= thr).astype(int)
    else:
        output["prediction"] = (output["probability"] >= 0.5).astype(int)
    duration_hours = output["duration_seconds"].sum() / 3600
    metrics = binary_metrics(
        output["label"].to_numpy(),
        output["probability"].to_numpy(),
        float(output["threshold"].iloc[0]),
        duration_hours,
    )
    metrics = _enrich_metrics(output, metrics, n_boot=n_boot)
    metrics["ensemble_checkpoints"] = [str(p) for p in checkpoints]
    metrics["ensemble_weights"] = weights
    metrics["tta_flip"] = bool(tta_flip)
    if smooth_seconds > 0:
        metrics["smooth_seconds"] = float(smooth_seconds)
        metrics["smooth_mode"] = smooth_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_ensemble" + (f"_smooth{int(smooth_seconds)}s" if smooth_seconds > 0 else "")
    output.to_csv(output_dir / f"{split}_predictions{suffix}.csv", index=False)
    (output_dir / f"{split}_metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
