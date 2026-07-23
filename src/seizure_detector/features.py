"""Multi-domain EEG features for clinical-translation-oriented models.

Implements recommendations from Moutonnet et al. (arXiv:2404.15332): combine
time-domain waveforms with spectral and network-domain representations.
All features are computed on-the-fly from cached bipolar windows (no EDF rebuild).
"""

from __future__ import annotations

import torch
from torch import nn

# Clinical EEG bands (Hz); windows are 0.5–45 Hz filtered at 128 Hz.
_BANDS = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
)


def bandpower_features(x: torch.Tensor, sample_rate: int = 128) -> torch.Tensor:
    """Per-channel log-bandpower → [B, C * n_bands]."""
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
    return torch.cat(feats, dim=-1)


def _band_analytic(x: torch.Tensor, sample_rate: int, low: float, high: float) -> torch.Tensor:
    """Band-limited analytic signal via FFT Hilbert (complex) → [B, C, T]."""
    n = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate).to(x.device)
    mask = (freqs >= low) & (freqs < high)
    spec = spec * mask.to(spec.dtype)
    # Hermitian construction for analytic signal via irfft of one-sided spectrum
    # is approximate; use full FFT Hilbert:
    full = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[1 : n // 2] = 2
        h[n // 2] = 1
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = torch.fft.ifft(full * h, dim=-1)
    # Apply band mask in frequency on the original then Hilbert
    band_x = torch.fft.irfft(spec, n=n, dim=-1)
    full_b = torch.fft.fft(band_x, dim=-1)
    return torch.fft.ifft(full_b * h, dim=-1)


def phase_locking_value(x: torch.Tensor, sample_rate: int = 128) -> torch.Tensor:
    """Upper-triangle mean PLV per band → [B, n_bands].

    Classic network-domain biomarker used in preictal literature (phase sync).
    """
    # x: [B, C, T]
    b, c, _ = x.shape
    outs = []
    # Pair indices for upper triangle
    ii, jj = torch.triu_indices(c, c, offset=1, device=x.device)
    for low, high in _BANDS:
        analytic = _band_analytic(x, sample_rate, low, high)
        phase = torch.angle(analytic)
        dphi = phase[:, ii] - phase[:, jj]  # [B, P, T]
        plv = torch.exp(1j * dphi).mean(dim=-1).abs()  # [B, P]
        outs.append(plv.mean(dim=-1))  # [B]
    return torch.stack(outs, dim=-1)


def spectral_connectivity(x: torch.Tensor, sample_rate: int = 128) -> torch.Tensor:
    """Mean pairwise magnitude-squared coherence per band → [B, n_bands]."""
    b, c, t = x.shape
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(t, d=1.0 / sample_rate).to(x.device)
    ii, jj = torch.triu_indices(c, c, offset=1, device=x.device)
    outs = []
    for low, high in _BANDS:
        mask = (freqs >= low) & (freqs < high)
        if not bool(mask.any()):
            outs.append(torch.zeros(b, device=x.device, dtype=x.dtype))
            continue
        s = spec[..., mask]  # [B, C, F]
        # Cross-spectrum
        pxx = (s.real.square() + s.imag.square()).mean(dim=-1)  # [B, C]
        # Average MSC over pairs and frequencies
        si = s[:, ii]
        sj = s[:, jj]
        cross = (si * sj.conj()).mean(dim=-1)
        num = cross.abs().square()
        den = (pxx[:, ii] * pxx[:, jj]).clamp_min(1e-12)
        msc = (num / den).clamp(0, 1).mean(dim=-1)
        outs.append(msc.real if torch.is_complex(msc) else msc)
    return torch.stack(outs, dim=-1)


class MultiDomainHead(nn.Module):
    """Project bandpower + PLV + coherence into a fused embedding."""

    def __init__(self, channels: int = 18, dim: int = 64, dropout: float = 0.35, sample_rate: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        band_dim = channels * len(_BANDS)
        net_dim = 2 * len(_BANDS)  # PLV + coherence
        self.band_proj = nn.Sequential(
            nn.Linear(band_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.net_proj = nn.Sequential(
            nn.Linear(net_dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = dim + dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.band_proj(bandpower_features(x, self.sample_rate))
        # PLV/coherence in float32 for numerical stability under AMP
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            plv = phase_locking_value(xf, self.sample_rate)
            coh = spectral_connectivity(xf, self.sample_rate)
            net = torch.cat([plv, coh], dim=-1).to(dtype=bands.dtype)
        net = self.net_proj(net)
        return torch.cat([bands, net], dim=-1)
