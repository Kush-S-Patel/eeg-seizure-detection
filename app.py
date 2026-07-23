"""Streamlit EEG monitoring and seizure-model dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from seizure_detector.audit import audit_dataset
from seizure_detector.config import OUTPUT_DIR, SignalConfig
from seizure_detector.dashboard import (
    band_powers,
    evaluation_curves,
    spectrogram_figure,
    stacked_eeg_figure,
)
from seizure_detector.labels import read_seizure_times
from seizure_detector.paths import load_splits, recording_paths
from seizure_detector.preprocess import MONTAGE_NAMES, extract_window, quality_metrics, recording_info

st.set_page_config(page_title="Neurotech EEG Monitor", page_icon="∿", layout="wide")
st.title("Neurotech EEG Seizure Research Monitor")
st.warning(
    "Research prototype only. Technician markers are weak labels and model outputs "
    "must not be used for diagnosis, treatment, or clinical monitoring."
)


@st.cache_data(show_spinner=False)
def get_splits() -> pd.DataFrame:
    return load_splits()


@st.cache_data(show_spinner=False)
def get_audit() -> pd.DataFrame:
    return audit_dataset(read_headers=False)


@st.cache_data(show_spinner=False, max_entries=24)
def get_window(path: str, start: float, duration: float, sample_rate: int):
    return extract_window(
        Path(path), start, duration, SignalConfig(sample_rate=sample_rate)
    )


def prediction_files() -> list[Path]:
    return sorted(OUTPUT_DIR.rglob("*_predictions.csv")) if OUTPUT_DIR.exists() else []


def page_overview() -> None:
    splits = get_splits()
    audit = get_audit()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected patients", splits["patient"].nunique())
    col2.metric("Selected recordings", len(splits))
    col3.metric("Local EDFs", int(audit["edf_exists"].sum()))
    col4.metric("Selected size", f"{splits['size_bytes'].sum() / 1024**3:.1f} GB")

    summary = (
        splits.groupby("split")
        .agg(
            patients=("patient", "nunique"),
            recordings=("record_id", "size"),
            seizure_recordings=("is_seizure", "sum"),
            bytes=("size_bytes", "sum"),
        )
        .reset_index()
    )
    summary["GB"] = summary.pop("bytes") / 1024**3
    local = audit.groupby("split")["edf_exists"].sum().rename("local_edfs")
    summary = summary.merge(local, on="split", how="left")
    st.subheader("Leakage-safe patient splits")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    figure = px.bar(
        summary,
        x="split",
        y=["recordings", "seizure_recordings", "local_edfs"],
        barmode="group",
        title="Selected recordings, weak-positive recordings, and local EDFs",
    )
    st.plotly_chart(figure, use_container_width=True)
    missing = audit[~audit["edf_exists"]]
    if len(missing):
        st.info(
            f"{len(missing)} validation/test EDFs are not local. Run "
            "`python scripts/downloader_100gb.py download --splits val test`, then audit again."
        )
    st.caption("Patient overlap across splits: none (enforced by the artifact loader).")


def _load_selected_record():
    splits = get_splits()
    audit = get_audit()
    local_ids = set(audit.loc[audit["edf_exists"], "record_id"])
    local = splits[splits["record_id"].isin(local_ids)].copy()
    split = st.sidebar.selectbox("Split", sorted(local["split"].unique()))
    local = local[local["split"] == split]
    patient = st.sidebar.selectbox("Patient", sorted(local["patient"].unique()))
    local = local[local["patient"] == patient]
    record_id = st.sidebar.selectbox("Recording", local["record_id"].tolist())
    paths = recording_paths(record_id)
    return local[local["record_id"] == record_id].iloc[0], paths


def page_browser() -> None:
    row, paths = _load_selected_record()
    info = recording_info(paths["edf"])
    duration = float(info["duration_seconds"])
    view_seconds = st.sidebar.slider("View length (seconds)", 5, 60, 20)
    start = st.sidebar.number_input(
        "Start time (seconds)",
        min_value=0.0,
        max_value=max(0.0, duration - view_seconds),
        value=0.0,
        step=float(view_seconds),
    )
    sample_rate = st.sidebar.select_slider("Display sample rate", [64, 128, 256], value=128)
    st.subheader(f"Patient {int(row.patient)} · session {int(row.session)}")
    st.caption(
        f"{duration / 3600:.2f} hours · {info['sample_rate']:.0f} Hz · "
        f"{len(info['channels'])} source channels"
    )
    with st.spinner("Reading selected EDF range…"):
        data, channel_mask = get_window(str(paths["edf"]), start, view_seconds, sample_rate)
    figure = stacked_eeg_figure(data, sample_rate, start_seconds=start)
    seizure_times = read_seizure_times(paths["xltek"], info["start"])
    for marker in seizure_times[(seizure_times >= start) & (seizure_times <= start + view_seconds)]:
        figure.add_vline(x=float(marker), line_dash="dash", line_color="red")
    st.plotly_chart(figure, use_container_width=True)
    missing = [name for name, present in zip(MONTAGE_NAMES, channel_mask, strict=True) if not present]
    if missing:
        st.warning(f"Unavailable montage derivations were zero-filled: {', '.join(missing)}")

    left, right = st.columns(2)
    channel = left.selectbox("Spectral channel", MONTAGE_NAMES)
    channel_idx = MONTAGE_NAMES.index(channel)
    left.plotly_chart(
        spectrogram_figure(data[channel_idx], sample_rate, f"{channel} spectrogram"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(band_powers(data[channel_idx], sample_rate), x="band", y="power", title="Band power"),
        use_container_width=True,
    )
    metrics = quality_metrics(data, sample_rate)
    cols = st.columns(3)
    cols[0].metric("Flat-channel fraction", f"{metrics['flat_channel_fraction']:.1%}")
    cols[1].metric("Clipped-sample fraction", f"{metrics['clipped_fraction']:.2%}")
    cols[2].metric("Scaled RMS", f"{metrics['rms']:.2f}")

    files = prediction_files()
    if files:
        selected = st.selectbox("Prediction timeline", files, format_func=lambda p: str(p.relative_to(OUTPUT_DIR)))
        predictions = pd.read_csv(selected)
        predictions = predictions[predictions["record_id"] == row.record_id]
        if len(predictions):
            risk = go.Figure(
                go.Scatter(
                    x=predictions["start_seconds"],
                    y=predictions["probability"],
                    mode="lines+markers",
                    name="Seizure risk",
                )
            )
            threshold = st.slider(
                "Display threshold",
                0.0,
                1.0,
                float(predictions["threshold"].iloc[0]),
                0.01,
            )
            risk.add_hline(y=threshold, line_dash="dash")
            risk.update_layout(xaxis_title="Recording time (s)", yaxis_title="Probability")
            st.plotly_chart(risk, use_container_width=True)


def page_model() -> None:
    files = prediction_files()
    if not files:
        st.info("No prediction files yet. Train and evaluate a checkpoint to populate this page.")
        return
    path = st.selectbox("Predictions", files, format_func=lambda p: str(p.relative_to(OUTPUT_DIR)))
    predictions = pd.read_csv(path)
    threshold = float(predictions["threshold"].iloc[0])
    st.plotly_chart(evaluation_curves(predictions), use_container_width=True)
    predictions["result"] = predictions.apply(
        lambda row: (
            "TP" if row.label == 1 and row.probability >= threshold else
            "FN" if row.label == 1 else
            "FP" if row.probability >= threshold else "TN"
        ),
        axis=1,
    )
    st.plotly_chart(
        px.histogram(
            predictions,
            x="probability",
            color="label",
            nbins=30,
            barmode="overlay",
            title="Probability calibration by weak label",
        ),
        use_container_width=True,
    )
    patient_summary = (
        predictions.groupby("patient")
        .agg(windows=("label", "size"), positives=("label", "sum"), mean_risk=("probability", "mean"))
        .reset_index()
    )
    st.subheader("Per-patient summary")
    st.dataframe(patient_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        predictions.to_csv(index=False),
        file_name=path.name,
        mime="text/csv",
    )
    metrics_path = path.with_name(path.name.replace("_predictions.csv", "_metrics.json"))
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    history = path.parent / "history.csv"
    if history.exists():
        table = pd.read_csv(history)
        metric_columns = [c for c in ("train_loss", "val_loss", "val_pr_auc") if c in table]
        st.plotly_chart(
            px.line(table, x="epoch", y=metric_columns, markers=True, title="Training history"),
            use_container_width=True,
        )


page = st.sidebar.radio("Page", ["Dataset overview", "EEG browser", "Model results"])
if page == "Dataset overview":
    page_overview()
elif page == "EEG browser":
    page_browser()
else:
    page_model()
