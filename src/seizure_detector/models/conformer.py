"""EEG-Conformer: CNN patch embed + Conformer blocks + bandpower fusion.

Targets ~200k–500k parameters for the Neurotech full-scale cache
([B, 18, 1280] bipolar windows at 128 Hz). Bandpower features are computed
on-the-fly from the cached waveform so no cache rebuild is required.
"""

from __future__ import annotations

import torch
from torch import nn

# Classic clinical EEG bands (Hz); windows are already 0.5–45 Hz filtered.
_BANDS = (
    (0.5, 4.0),   # delta
    (4.0, 8.0),   # theta
    (8.0, 13.0),  # alpha
    (13.0, 30.0), # beta
    (30.0, 45.0), # gamma
)


def bandpower_features(x: torch.Tensor, sample_rate: int = 128) -> torch.Tensor:
    """Per-channel log-bandpower for each band → [B, channels * n_bands]."""
    # x: [B, C, T]
    spectrum = torch.fft.rfft(x, dim=-1)
    power = spectrum.real.square() + spectrum.imag.square()
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate).to(x.device)
    feats = []
    for low, high in _BANDS:
        mask = (freqs >= low) & (freqs < high)
        if not bool(mask.any()):
            feats.append(torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype))
            continue
        band = power[..., mask].mean(dim=-1).clamp_min(1e-8).log()
        feats.append(torch.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.cat(feats, dim=-1)  # [B, C * n_bands]


class _FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConformerBlock(nn.Module):
    """Macaron-style FFN → MHSA → depthwise conv → FFN."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, conv_kernel: int = 15):
        super().__init__()
        self.ff1 = _FeedForward(dim, expansion=2, dropout=dropout)
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_conv = nn.LayerNorm(dim)
        padding = conv_kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=padding, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ff2 = _FeedForward(dim, expansion=2, dropout=dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout_attn(attn_out)
        h = self.norm_conv(x).transpose(1, 2)  # [B, D, T]
        x = x + self.conv(h).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class EEGConformer(nn.Module):
    def __init__(
        self,
        channels: int = 18,
        samples: int = 1280,
        dropout: float = 0.35,
        dim: int = 96,
        heads: int = 4,
        n_blocks: int = 3,
        sample_rate: int = 128,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        # Stride-8 patch embed: 1280 → 160 tokens.
        self.embed = nn.Sequential(
            nn.Conv1d(channels, dim, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [_ConformerBlock(dim, heads=heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.pool_attn = nn.Linear(dim, 1)
        band_dim = channels * len(_BANDS)
        self.band_proj = nn.Sequential(
            nn.Linear(band_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)
        bands = self.band_proj(bandpower_features(x, self.sample_rate))
        tokens = self.embed(x).transpose(1, 2)  # [B, T', D]
        for block in self.blocks:
            tokens = block(tokens)
        weights = torch.softmax(self.pool_attn(tokens), dim=1)
        pooled = (tokens * weights).sum(dim=1)
        fused = torch.cat([pooled, bands], dim=-1)
        return self.classifier(fused).squeeze(1)


class EEGConformerMultiDomain(nn.Module):
    """Conformer + spectral/network fusion (Moutonnet et al. multi-domain guidance)."""

    def __init__(
        self,
        channels: int = 18,
        samples: int = 1280,
        dropout: float = 0.35,
        dim: int = 96,
        heads: int = 4,
        n_blocks: int = 3,
        sample_rate: int = 128,
    ):
        super().__init__()
        from ..features import MultiDomainHead

        self.sample_rate = sample_rate
        self.embed = nn.Sequential(
            nn.Conv1d(channels, dim, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [_ConformerBlock(dim, heads=heads, dropout=dropout) for _ in range(n_blocks)]
        )
        self.pool_attn = nn.Linear(dim, 1)
        self.domain = MultiDomainHead(channels=channels, dim=dim, dropout=dropout, sample_rate=sample_rate)
        self.classifier = nn.Sequential(
            nn.Linear(dim + self.domain.out_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)
        domain = self.domain(x)
        tokens = self.embed(x).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        weights = torch.softmax(self.pool_attn(tokens), dim=1)
        pooled = (tokens * weights).sum(dim=1)
        fused = torch.cat([pooled, domain], dim=-1)
        return self.classifier(fused).squeeze(1)


def EEGConformerLarge(
    channels: int = 18,
    samples: int = 1280,
    dropout: float = 0.4,
    sample_rate: int = 128,
) -> EEGConformer:
    """~1.2M-param Conformer for full-scale training on an 8GB GPU."""
    return EEGConformer(
        channels=channels,
        samples=samples,
        dropout=dropout,
        dim=128,
        heads=4,
        n_blocks=4,
        sample_rate=sample_rate,
    )
