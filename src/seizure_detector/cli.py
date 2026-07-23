"""Command-line workflow for auditing, preparation, training, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_dataset, print_report, save_report
from .cache import build_window_cache
from .config import (
    OUTPUT_DIR,
    SignalConfig,
    TrainConfig,
    WINDOW_CACHE_DIR,
    WINDOWS_PATH,
    WindowConfig,
)
from .dataset import load_windows
from .engine import evaluate_checkpoint, predict_windows, train_model
from .forecast import FORECAST_WINDOWS_PATH, ForecastConfig, build_forecast_windows
from .models import MODELS
from .windows import build_window_manifest


def _audit(args) -> int:
    report = audit_dataset(read_headers=not args.fast)
    save_report(report)
    return 0 if print_report(report) else 1


def _prepare(args) -> int:
    config = WindowConfig(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        positive_radius_seconds=args.positive_radius,
        negative_guard_seconds=args.negative_guard,
        max_negative_windows_per_recording=args.max_negatives,
        seed=args.seed,
    )
    build_window_manifest(
        config=config,
        include_splits=tuple(args.splits),
        max_records=args.max_records,
    )
    return 0


def _cache(args) -> int:
    build_window_cache(
        load_windows(),
        SignalConfig(sample_rate=args.sample_rate),
        Path(args.cache_dir),
        force=args.force,
    )
    return 0


def _prepare_forecast(args) -> int:
    cfg = ForecastConfig(
        sop_seconds=args.sop_seconds,
        sph_seconds=args.sph_seconds,
        peri_ictal_guard_seconds=args.peri_guard,
        post_ictal_guard_seconds=args.post_guard,
        interictal_guard_seconds=args.interictal_guard,
        max_negatives_per_recording=args.max_negatives,
        seed=args.seed,
        include_seizure_free_negatives=not args.marker_records_only,
        first_onset_only=args.first_onset_only,
        min_inter_onset_gap_seconds=args.min_inter_onset_gap,
    )
    table = build_forecast_windows(
        load_windows(Path(args.detection_windows)),
        cfg=cfg,
        output_path=Path(args.output),
    )
    print(
        f"Wrote {len(table):,} forecast windows "
        f"(pos={int(table.label.sum()):,}, prev={table.label.mean():.4f}) -> {args.output}"
    )
    return 0


def _train(args) -> int:
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.workers,
        seed=args.seed,
        max_train_windows=args.max_train_windows,
        device=args.device,
        model_name=args.model,
        dropout=args.dropout,
        use_balanced_sampler=not args.no_balanced_sampler,
        pos_weight=args.pos_weight,
        auto_pos_weight=args.auto_pos_weight,
        loss_fn=args.loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        augment_noise_std=args.augment_noise_std,
        augment_channel_dropout=args.augment_channel_dropout,
        augment_time_flip=args.augment_time_flip,
        mixup_alpha=args.mixup_alpha,
        lr_scheduler=args.lr_scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_delta=args.min_delta,
        ema_decay=args.ema_decay,
        resume_from=args.resume,
        record_norm=args.record_norm,
    )
    windows = load_windows(Path(args.windows)) if args.windows else load_windows()
    checkpoint = train_model(
        windows,
        Path(args.output),
        train_config=cfg,
        signal_config=SignalConfig(sample_rate=args.sample_rate),
        cache_dir=Path(args.cache_dir),
        rebuild_cache=args.rebuild_cache,
    )
    print(f"Best checkpoint: {checkpoint}")
    return 0


def _evaluate(args) -> int:
    from .engine import evaluate_ensemble

    windows = load_windows(Path(args.windows)) if args.windows else load_windows()
    if args.ensemble:
        checkpoints = [Path(args.checkpoint), *[Path(p) for p in args.ensemble]]
        weights = None
        if args.ensemble_weights:
            weights = [float(w) for w in args.ensemble_weights]
            if len(weights) != len(checkpoints):
                raise SystemExit(
                    f"--ensemble-weights needs {len(checkpoints)} values (got {len(weights)})"
                )
        metrics = evaluate_ensemble(
            checkpoints,
            windows,
            args.split,
            Path(args.output),
            weights=weights,
            smooth_seconds=args.smooth_seconds,
            smooth_mode=args.smooth_mode,
            tta_flip=not args.no_tta,
        )
    else:
        metrics = evaluate_checkpoint(
            Path(args.checkpoint),
            windows,
            args.split,
            Path(args.output),
            smooth_seconds=args.smooth_seconds,
            smooth_mode=args.smooth_mode,
            tta_flip=not args.no_tta,
        )
    print(json.dumps(metrics, indent=2))
    return 0


def _predict(args) -> int:
    windows = load_windows(Path(args.windows)) if args.windows else load_windows()
    table = windows[windows["split"] == args.split].reset_index(drop=True)
    predictions = predict_windows(
        Path(args.checkpoint),
        table,
        smooth_seconds=args.smooth_seconds,
        smooth_mode=args.smooth_mode,
        tta_flip=not args.no_tta,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seizure-detector")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Validate selected local EDFs")
    audit.add_argument("--fast", action="store_true")
    audit.set_defaults(func=_audit)

    prepare = sub.add_parser("prepare", help="Create weak-label window metadata")
    prepare.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    prepare.add_argument("--window-seconds", type=float, default=10)
    prepare.add_argument("--stride-seconds", type=float, default=5)
    prepare.add_argument("--positive-radius", type=float, default=30)
    prepare.add_argument("--negative-guard", type=float, default=300)
    prepare.add_argument("--max-negatives", type=int, default=200)
    prepare.add_argument("--max-records", type=int)
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.set_defaults(func=_prepare)

    forecast = sub.add_parser(
        "prepare-forecast",
        help="Re-label cached windows for preictal seizure forecasting (SOP/SPH)",
    )
    forecast.add_argument(
        "--detection-windows",
        default=str(WINDOWS_PATH),
        help="Detection windows.parquet to re-label",
    )
    forecast.add_argument("--output", default=str(FORECAST_WINDOWS_PATH))
    forecast.add_argument("--sop-seconds", type=float, default=1800.0, help="Seizure occurrence period")
    forecast.add_argument("--sph-seconds", type=float, default=300.0, help="Seizure prediction horizon")
    forecast.add_argument("--peri-guard", type=float, default=300.0)
    forecast.add_argument("--post-guard", type=float, default=1800.0)
    forecast.add_argument("--interictal-guard", type=float, default=7200.0)
    forecast.add_argument("--max-negatives", type=int, default=200)
    forecast.add_argument("--seed", type=int, default=1337)
    forecast.add_argument(
        "--marker-records-only",
        action="store_true",
        help="Skip seizure-free recordings as interictal negatives",
    )
    forecast.add_argument(
        "--first-onset-only",
        action="store_true",
        help="Only use the first estimated onset per recording",
    )
    forecast.add_argument(
        "--min-inter-onset-gap",
        type=float,
        default=0.0,
        help="Drop onsets closer than this many seconds to a prior onset",
    )
    forecast.set_defaults(func=_prepare_forecast)

    cache = sub.add_parser(
        "cache",
        help="Precompute filtered/resampled window arrays (run once after prepare)",
    )
    cache.add_argument("--sample-rate", type=int, default=128)
    cache.add_argument("--cache-dir", default=str(WINDOW_CACHE_DIR))
    cache.add_argument("--force", action="store_true", help="Rebuild even if cache looks valid")
    cache.set_defaults(func=_cache)

    train = sub.add_parser("train", help="Train a seizure-window classifier")
    train.add_argument("--output", default=str(OUTPUT_DIR / "baseline"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=7, help="Epochs without improvement before stopping")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--max-train-windows", type=int)
    train.add_argument("--sample-rate", type=int, default=128)
    train.add_argument("--device", default="auto")
    train.add_argument("--cache-dir", default=str(WINDOW_CACHE_DIR))
    train.add_argument(
        "--rebuild-cache", action="store_true", help="Force-rebuild the window cache first"
    )
    train.add_argument(
        "--windows",
        default=None,
        help="Window parquet path (default: detection windows; use forecast parquet for prediction)",
    )

    # Model
    train.add_argument("--model", choices=sorted(MODELS), default="eegnet_deep")
    train.add_argument("--dropout", type=float, default=0.35)

    # Class imbalance
    train.add_argument(
        "--no-balanced-sampler", action="store_true",
        help="Disable oversampling of positive windows during training",
    )
    train.add_argument(
        "--pos-weight", type=float, default=None,
        help="Fixed positive-class weight for the loss (overrides --auto-pos-weight)",
    )
    train.add_argument(
        "--auto-pos-weight", action="store_true",
        help="Derive the loss positive-class weight from the train split's negative:positive ratio",
    )

    # Loss / regularization
    train.add_argument("--loss", choices=["bce", "focal"], default="bce")
    train.add_argument("--focal-gamma", type=float, default=2.0)
    train.add_argument("--label-smoothing", type=float, default=0.0)
    train.add_argument(
        "--grad-clip-norm", type=float, default=1.0,
        help="Max gradient norm; set to 0 to disable clipping",
    )
    train.add_argument("--augment-noise-std", type=float, default=0.0)
    train.add_argument("--augment-channel-dropout", type=float, default=0.0)
    train.add_argument(
        "--augment-time-flip",
        type=float,
        default=0.0,
        help="Probability of reversing each window along time during training",
    )
    train.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Beta(alpha,alpha) mixup; 0 disables",
    )
    train.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for eval/checkpoint weights; 0 disables (try 0.999)",
    )
    train.add_argument(
        "--resume",
        default=None,
        help="Warm-start model weights from an existing checkpoint",
    )
    train.add_argument(
        "--record-norm",
        action="store_true",
        help="Apply per-recording robust scale using window_cache/record_stats.npz",
    )

    # LR schedule
    train.add_argument("--lr-scheduler", choices=["none", "plateau", "cosine", "onecycle"], default="plateau")
    train.add_argument("--scheduler-patience", type=int, default=2)
    train.add_argument("--scheduler-factor", type=float, default=0.5)
    train.add_argument("--min-lr", type=float, default=1e-6)

    # Early stopping
    train.add_argument(
        "--early-stopping-metric", choices=["pr_auc", "roc_auc", "f1", "loss"], default="pr_auc"
    )
    train.add_argument("--min-delta", type=float, default=0.0)
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate a held-out split")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--output", default=str(OUTPUT_DIR / "baseline"))
    evaluate.add_argument(
        "--windows",
        default=None,
        help="Window parquet path (default: detection windows)",
    )
    evaluate.add_argument(
        "--smooth-seconds",
        type=float,
        default=0.0,
        help="Pool overlapping-window probabilities within ±this many seconds (0=off)",
    )
    evaluate.add_argument(
        "--smooth-mode",
        choices=["softmax", "max", "mean"],
        default="softmax",
        help="Temporal pooling mode when --smooth-seconds > 0",
    )
    evaluate.add_argument(
        "--ensemble",
        nargs="+",
        default=[],
        help="Additional checkpoints to blend with the primary checkpoint",
    )
    evaluate.add_argument(
        "--ensemble-weights",
        nargs="+",
        default=None,
        help="Blend weights for primary + --ensemble checkpoints (same order)",
    )
    evaluate.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable time-flip test-time augmentation",
    )
    evaluate.set_defaults(func=_evaluate)

    predict = sub.add_parser("predict", help="Write per-window probabilities")
    predict.add_argument("checkpoint")
    predict.add_argument("--split", choices=["train", "val", "test"], default="test")
    predict.add_argument("--output", default=str(OUTPUT_DIR / "predictions.csv"))
    predict.add_argument("--windows", default=None)
    predict.add_argument(
        "--smooth-seconds",
        type=float,
        default=0.0,
        help="Pool overlapping-window probabilities within ±this many seconds (0=off)",
    )
    predict.add_argument(
        "--smooth-mode",
        choices=["softmax", "max", "mean"],
        default="softmax",
    )
    predict.add_argument("--no-tta", action="store_true")
    predict.set_defaults(func=_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
