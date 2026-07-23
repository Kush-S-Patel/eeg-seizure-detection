"""Compact EEGNet-style baseline for 18-channel time windows."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet1D(nn.Module):
    def __init__(self, channels: int = 18, samples: int = 1280, dropout: float = 0.35):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 31), padding=(0, 15), groups=16, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(32 * 8, 1)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return self.classifier(x.flatten(1)).squeeze(1)
