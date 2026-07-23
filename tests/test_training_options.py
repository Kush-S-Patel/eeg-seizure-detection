import pandas as pd
import torch

from seizure_detector.config import TrainConfig
from seizure_detector.engine import _augment, _build_scheduler, _score
from seizure_detector.losses import FocalLossWithLogits, build_loss_fn, resolve_pos_weight, smooth_targets


def test_resolve_pos_weight_explicit_wins():
    labels = pd.Series([0, 0, 0, 1])
    assert resolve_pos_weight(labels, pos_weight=2.5, auto_pos_weight=True) == 2.5


def test_resolve_pos_weight_auto_matches_class_ratio():
    labels = pd.Series([0] * 8 + [1] * 2)
    weight = resolve_pos_weight(labels, pos_weight=None, auto_pos_weight=True)
    assert weight == 4.0


def test_resolve_pos_weight_auto_is_capped():
    labels = pd.Series([0] * 1000 + [1])
    weight = resolve_pos_weight(labels, pos_weight=None, auto_pos_weight=True)
    assert weight == 15.0


def test_resolve_pos_weight_disabled_by_default():
    labels = pd.Series([0, 0, 1])
    assert resolve_pos_weight(labels, pos_weight=None, auto_pos_weight=False) is None


def test_focal_loss_is_finite_and_positive():
    loss_fn = FocalLossWithLogits(gamma=2.0, pos_weight=3.0)
    logits = torch.tensor([2.0, -2.0, 0.1, -0.1])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_focal_loss_downweights_easy_examples():
    loss_fn = FocalLossWithLogits(gamma=2.0)
    easy = loss_fn(torch.tensor([5.0]), torch.tensor([1.0]))
    hard = loss_fn(torch.tensor([-5.0]), torch.tensor([1.0]))
    assert easy.item() < hard.item()


def test_build_loss_fn_variants():
    bce = build_loss_fn("bce", pos_weight=None, focal_gamma=2.0, device=torch.device("cpu"))
    focal = build_loss_fn("focal", pos_weight=2.0, focal_gamma=2.0, device=torch.device("cpu"))
    assert isinstance(bce, torch.nn.BCEWithLogitsLoss)
    assert isinstance(focal, FocalLossWithLogits)


def test_smooth_targets_pulls_toward_half():
    y = torch.tensor([0.0, 1.0])
    smoothed = smooth_targets(y, 0.2)
    assert torch.allclose(smoothed, torch.tensor([0.1, 0.9]))
    assert torch.equal(smooth_targets(y, 0.0), y)


def test_score_prefers_metric_when_available():
    metrics = {"pr_auc": 0.42, "roc_auc": 0.7}
    assert _score("pr_auc", metrics, loss=1.0) == 0.42
    assert _score("loss", metrics, loss=0.5) == -0.5
    assert _score("pr_auc", None, loss=0.5) == -0.5


def test_augment_channel_dropout_zeros_some_channels():
    torch.manual_seed(0)
    cfg = TrainConfig(augment_channel_dropout=1.0, augment_noise_std=0.0)
    x = torch.ones(2, 4, 8)
    out = _augment(x, cfg)
    assert torch.all(out == 0)


def test_augment_noise_changes_values():
    torch.manual_seed(0)
    cfg = TrainConfig(augment_noise_std=1.0, augment_channel_dropout=0.0)
    x = torch.zeros(4, 4, 16)
    out = _augment(x, cfg)
    assert not torch.allclose(out, x)


def test_build_scheduler_none_returns_none():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = TrainConfig(lr_scheduler="none")
    assert _build_scheduler(optimizer, cfg, steps_per_epoch=10) is None


def test_build_scheduler_plateau_and_cosine():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    plateau = _build_scheduler(optimizer, TrainConfig(lr_scheduler="plateau"), steps_per_epoch=10)
    assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)

    optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cosine = _build_scheduler(optimizer2, TrainConfig(lr_scheduler="cosine", epochs=5), steps_per_epoch=10)
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
