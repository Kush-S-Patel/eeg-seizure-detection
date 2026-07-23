"""Loss functions and class-imbalance helpers for binary window classification."""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn


class FocalLossWithLogits(nn.Module):
    """Binary focal loss (Lin et al. 2017) with an optional positive-class weight.

    Down-weights easy, already-confident examples so training spends more
    gradient budget on hard/rare positives (seizures) instead of the vast
    majority of easy negatives — an alternative to (or complement of)
    oversampling for a ~1-15% positive rate.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t).clamp(min=0).pow(self.gamma)
        if self.pos_weight is not None:
            class_weight = targets * self.pos_weight + (1 - targets)
        else:
            class_weight = 1.0
        return (modulating * class_weight * bce).mean()


def resolve_pos_weight(
    train_labels: pd.Series,
    *,
    pos_weight: float | None,
    auto_pos_weight: bool,
) -> float | None:
    """Pick a positive-class weight for the loss.

    Explicit ``pos_weight`` wins. Otherwise, if ``auto_pos_weight`` is set,
    derive it from the observed negative:positive ratio in the training
    table so the loss compensates for class imbalance directly (as an
    alternative to, or combined with, the balanced sampler).
    """
    if pos_weight is not None:
        return pos_weight
    if not auto_pos_weight:
        return None
    counts = train_labels.value_counts()
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    if positives == 0:
        return None
    # Cap so focal+auto_pos_weight does not explode gradients on rare positives.
    return float(min(max(negatives / positives, 1.0), 15.0))


def build_loss_fn(
    loss_name: str,
    *,
    pos_weight: float | None,
    focal_gamma: float,
    device: torch.device,
) -> nn.Module:
    if loss_name == "bce":
        weight_tensor = torch.tensor(pos_weight, device=device) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    if loss_name == "focal":
        return FocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight)
    raise ValueError(f"Unknown loss_fn {loss_name!r}; choose 'bce' or 'focal'")


def smooth_targets(targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    """Pull hard 0/1 targets slightly toward 0.5 to discourage overconfidence."""
    if label_smoothing <= 0:
        return targets
    return targets * (1 - label_smoothing) + 0.5 * label_smoothing
