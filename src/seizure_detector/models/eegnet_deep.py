"""A wider/deeper EEGNet-style variant for when the compact baseline underfits."""

from __future__ import annotations

import torch
from torch import nn


class EEGNetDeep(nn.Module):
    def __init__(
        self,
        channels: int = 18,
        samples: int = 1280,
        dropout: float = 0.4,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
    ):
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters, spatial_filters, kernel_size=(channels, 1),
                groups=temporal_filters, bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable1 = nn.Sequential(
            nn.Conv2d(
                spatial_filters, spatial_filters, kernel_size=(1, 31), padding=(0, 15),
                groups=spatial_filters, bias=False,
            ),
            nn.Conv2d(spatial_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable2 = nn.Sequential(
            nn.Conv2d(
                separable_filters, separable_filters, kernel_size=(1, 15), padding=(0, 7),
                groups=separable_filters, bias=False,
            ),
            nn.Conv2d(separable_filters, separable_filters * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters * 2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(separable_filters * 2 * 8, 1)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable1(x)
        x = self.separable2(x)
        return self.classifier(x.flatten(1)).squeeze(1)
