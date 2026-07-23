"""Model registry."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .cnn_gru import CNNGRU
from .conformer import EEGConformer, EEGConformerLarge, EEGConformerMultiDomain
from .eegnet import EEGNet1D
from .eegnet_deep import EEGNetDeep

MODELS: dict[str, Callable[..., nn.Module]] = {
    "eegnet": EEGNet1D,
    "eegnet_deep": EEGNetDeep,
    "cnn_gru": CNNGRU,
    "eeg_conformer": EEGConformer,
    "eeg_conformer_large": EEGConformerLarge,
    "eeg_conformer_multidomain": EEGConformerMultiDomain,
}


def create_model(name: str = "eegnet", **kwargs) -> nn.Module:
    try:
        return MODELS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}") from exc


def register_model(name: str, constructor: Callable[..., nn.Module]) -> None:
    """Register an experimental architecture without changing training code."""
    if name in MODELS:
        raise ValueError(f"Model {name!r} is already registered")
    MODELS[name] = constructor


# IMPROVEMENT: register channel-attention, temporal transformer, or pretrained
# encoders here while retaining the same logits output contract.
