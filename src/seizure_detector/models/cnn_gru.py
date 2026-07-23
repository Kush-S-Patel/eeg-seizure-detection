"""Temporal-CNN front end + attention-pooled BiGRU.

An alternative to the pure-convolutional EEGNet architectures: the GRU gives
the model an explicit notion of temporal order/evolution across the window,
which point-marker seizure activity (a build-up/spread pattern, not just a
static spectral signature) can benefit from.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNGRU(nn.Module):
    def __init__(
        self,
        channels: int = 18,
        samples: int = 1280,
        dropout: float = 0.35,
        conv_channels: int = 32,
        gru_hidden: int = 64,
    ):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(channels, conv_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(conv_channels * 2),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(conv_channels * 2, gru_hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(gru_hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)
        features = self.frontend(x)  # [B, C, T']
        features = features.transpose(1, 2)  # [B, T', C]
        sequence, _ = self.gru(features)  # [B, T', 2H]
        weights = torch.softmax(self.attention(sequence), dim=1)  # [B, T', 1]
        pooled = (sequence * weights).sum(dim=1)  # [B, 2H]
        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(1)
