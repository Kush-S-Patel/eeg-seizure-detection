"""Plot and summary helpers used by the Streamlit application."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid
from scipy.signal import spectrogram, welch
from sklearn.metrics import precision_recall_curve, roc_curve

from .preprocess import MONTAGE_NAMES


def stacked_eeg_figure(
    data: np.ndarray,
    sample_rate: float,
    *,
    start_seconds: float = 0,
    names: tuple[str, ...] = MONTAGE_NAMES,
) -> go.Figure:
    time = start_seconds + np.arange(data.shape[-1]) / sample_rate
    spacing = 2.5 * max(float(np.nanstd(data)), 1.0)
    figure = go.Figure()
    for index, name in enumerate(names[: data.shape[0]]):
        offset = (data.shape[0] - index - 1) * spacing
        figure.add_trace(
            go.Scattergl(
                x=time,
                y=data[index] + offset,
                mode="lines",
                name=name,
                line={"width": 0.8},
                hovertemplate=f"{name}<br>t=%{{x:.2f}}s<extra></extra>",
            )
        )
    figure.update_layout(
        height=max(500, 35 * data.shape[0]),
        xaxis_title="Recording time (s)",
        yaxis_title="Bipolar channels (robust-scaled, offset)",
        showlegend=False,
        margin={"l": 30, "r": 20, "t": 20, "b": 35},
    )
    return figure


def spectrogram_figure(signal: np.ndarray, sample_rate: float, title: str) -> go.Figure:
    frequencies, times, power = spectrogram(
        signal, fs=sample_rate, nperseg=min(int(sample_rate * 2), len(signal))
    )
    keep = frequencies <= 45
    figure = go.Figure(
        go.Heatmap(
            x=times,
            y=frequencies[keep],
            z=10 * np.log10(power[keep] + 1e-12),
            colorscale="Viridis",
            colorbar={"title": "dB"},
        )
    )
    figure.update_layout(title=title, xaxis_title="Window time (s)", yaxis_title="Hz")
    return figure


def band_powers(signal: np.ndarray, sample_rate: float) -> pd.DataFrame:
    frequencies, density = welch(signal, fs=sample_rate, nperseg=min(len(signal), 512))
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    rows = []
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        rows.append({"band": name, "power": float(trapezoid(density[mask], frequencies[mask]))})
    return pd.DataFrame(rows)


def evaluation_curves(predictions: pd.DataFrame) -> go.Figure:
    targets = predictions["label"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Precision–recall", "ROC"))
    if len(np.unique(targets)) == 2:
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        fpr, tpr, _ = roc_curve(targets, probabilities)
        figure.add_trace(go.Scatter(x=recall, y=precision, name="PR"), row=1, col=1)
        figure.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC"), row=1, col=2)
    figure.update_xaxes(title_text="Recall", row=1, col=1)
    figure.update_yaxes(title_text="Precision", row=1, col=1)
    figure.update_xaxes(title_text="False-positive rate", row=1, col=2)
    figure.update_yaxes(title_text="True-positive rate", row=1, col=2)
    figure.update_layout(height=420, showlegend=False)
    return figure
