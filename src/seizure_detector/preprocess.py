"""Lazy EDF window loading and a robust longitudinal bipolar montage."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from .config import SignalConfig

BIPOLAR_PAIRS = (
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("FZ", "CZ"), ("CZ", "PZ"),
)
MONTAGE_NAMES = tuple(f"{a}-{b}" for a, b in BIPOLAR_PAIRS)
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def canonical_channel(name: str) -> str:
    value = name.upper().replace("EEG", "").replace("REF", "")
    value = re.sub(r"[^A-Z0-9]", "", value)
    return ALIASES.get(value, value)


@lru_cache(maxsize=8)
def open_raw(path: str):
    import mne

    return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


def _edf_field(raw: bytes, start: int, length: int) -> str:
    return raw[start : start + length].decode("ascii", errors="replace").strip()


def _edf_int_field(raw: bytes, start: int, length: int) -> int | None:
    text = _edf_field(raw, start, length)
    if not text or text == "-1":
        return None
    return int(text)


def _parse_edf_start(data: bytes) -> pd.Timestamp:
    if len(data) < 184:
        raise ValueError("EDF start timestamp unavailable")
    date_text = _edf_field(data, 168, 8)
    time_text = _edf_field(data, 176, 8)
    day, month, year2 = date_text.split(".")
    hour, minute, second = time_text.split(".")
    year = 2000 + int(year2) if int(year2) < 85 else 1900 + int(year2)
    return pd.Timestamp(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
    )


def recording_info_from_bids(edf_path: Path) -> dict | None:
    """Use tiny BIDS sidecars for EDF+C recordings (common in this dataset)."""
    json_path = Path(str(edf_path).replace("_eeg.edf", "_eeg.json"))
    if not json_path.is_file():
        return None
    import json

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    duration = meta.get("RecordingDuration")
    if duration is None or duration == "":
        return None
    start = None
    if edf_path.is_file() and edf_path.stat().st_size >= 184:
        try:
            start = _parse_edf_start(edf_path.read_bytes()[:256])
        except ValueError:
            start = None
    sfreq = meta.get("SamplingFrequency")
    if sfreq is None or sfreq == "":
        sfreq = 256.0
    return {
        "duration_seconds": float(duration),
        "sample_rate": float(sfreq),
        "channels": [],
        "start": start,
    }


def parse_edf_header(path: Path | str, *, file_size_bytes: int | None = None) -> dict:
    """Read duration/channels/start from an EDF header without loading signal data."""
    data = Path(path).read_bytes()
    if len(data) < 256:
        raise ValueError(f"EDF header too short ({len(data)} bytes)")

    header_bytes = int(_edf_field(data, 184, 8))
    num_records = _edf_int_field(data, 200, 8)
    record_duration = float(_edf_field(data, 208, 8) or "0")
    num_signals = int(_edf_field(data, 216, 8))
    if len(data) < header_bytes:
        raise ValueError(
            f"EDF header incomplete: need {header_bytes} bytes, have {len(data)} "
            "(re-fetch a larger header stub)"
        )
    if record_duration <= 0:
        raise ValueError("EDF header missing record duration")

    channels: list[str] = []
    samples_per_record: list[int] = []
    for index in range(num_signals):
        block = data[256 + index * 256 : 256 + (index + 1) * 256]
        label = _edf_field(block, 0, 16)
        per_record = int(_edf_field(block, 216, 8))
        channels.append(label)
        samples_per_record.append(per_record)

    sample_rate = samples_per_record[0] / record_duration if samples_per_record else 0.0
    if num_records is None:
        total_size = file_size_bytes if file_size_bytes is not None else Path(path).stat().st_size
        bytes_per_record = sum(samples_per_record) * 2
        if bytes_per_record <= 0:
            raise ValueError("Cannot infer EDF+C duration without signal sample counts")
        num_records = (total_size - header_bytes) // bytes_per_record
        if num_records <= 0:
            raise ValueError("Cannot infer EDF+C duration from file size")

    start = _parse_edf_start(data[:256])

    return {
        "duration_seconds": num_records * record_duration,
        "sample_rate": float(sample_rate),
        "channels": channels,
        "start": start,
    }


def recording_info(path: Path, *, file_size_bytes: int | None = None) -> dict:
    path = Path(path)
    bids = recording_info_from_bids(path)
    if bids is not None:
        return bids
    size = path.stat().st_size
    # Header stubs from Phase A are a few KB; MNE cannot parse truncated EDFs.
    if size < 10 * 1024 * 1024:
        return parse_edf_header(path, file_size_bytes=file_size_bytes)
    raw = open_raw(str(path))
    return {
        "duration_seconds": raw.n_times / float(raw.info["sfreq"]),
        "sample_rate": float(raw.info["sfreq"]),
        "channels": list(raw.ch_names),
        "start": raw.info.get("meas_date"),
    }


def resolve_montage_channels(ch_names: list[str]) -> tuple[dict[str, int], list[str]]:
    """Map canonical channel names to source indices for the montage this file supports."""
    channel_index = {canonical_channel(name): idx for idx, name in enumerate(ch_names)}
    present = sorted({name for pair in BIPOLAR_PAIRS for name in pair if name in channel_index})
    return channel_index, present


def build_bipolar_montage(
    by_name: dict[str, np.ndarray], expected_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-channel arrays into the 18-derivation montage plus presence mask."""
    montage: list[np.ndarray] = []
    mask: list[float] = []
    for left, right in BIPOLAR_PAIRS:
        if left in by_name and right in by_name:
            signal = by_name[left] - by_name[right]
            mask.append(1.0)
        else:
            signal = np.zeros(expected_len, dtype=np.float64)
            mask.append(0.0)
        if len(signal) < expected_len:
            signal = np.pad(signal, (0, expected_len - len(signal)))
        montage.append(signal[:expected_len])
    return np.asarray(montage), np.asarray(mask, dtype=np.float32)


def _bandpass(data: np.ndarray, sample_rate: float, low: float, high: float) -> np.ndarray:
    nyquist = sample_rate / 2
    high = min(high, nyquist * 0.95)
    if low <= 0 or low >= high:
        return data
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def filter_and_resample(data: np.ndarray, source_rate: float, config: SignalConfig) -> np.ndarray:
    """Bandpass then resample to the target rate. Cheapest when run once per file."""
    data = _bandpass(data, source_rate, config.low_hz, config.high_hz)
    if int(round(source_rate)) != config.sample_rate:
        data = resample_poly(data, config.sample_rate, int(round(source_rate)), axis=-1)
    return data


def robust_scale(data: np.ndarray, clip: float = 8.0) -> np.ndarray:
    median = np.median(data, axis=-1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=-1, keepdims=True)
    scaled = (data - median) / np.maximum(1.4826 * mad, 1e-8)
    return np.clip(scaled, -clip, clip).astype(np.float32)


def extract_window(
    edf_path: Path,
    start_seconds: float,
    duration_seconds: float,
    config: SignalConfig = SignalConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return [18, samples] bipolar data and a present-channel mask.

    Reads, filters, and resamples only this single window. Fine for one-off
    exploration (the dashboard), but training/evaluation should use the
    precomputed array cache in ``seizure_detector.cache`` instead of calling
    this per-sample, since redoing the filter+resample per window per epoch is
    the main reason training was slow.
    """
    raw = open_raw(str(edf_path))
    source_rate = float(raw.info["sfreq"])
    start = max(0, int(round(start_seconds * source_rate)))
    stop = min(raw.n_times, start + int(round(duration_seconds * source_rate)))
    channel_index, present = resolve_montage_channels(raw.ch_names)
    picks = [channel_index[name] for name in present]
    values = raw.get_data(picks=picks, start=start, stop=stop)
    by_name = dict(zip(present, values, strict=True))
    expected = int(round(duration_seconds * source_rate))

    montage, mask = build_bipolar_montage(by_name, expected)
    data = filter_and_resample(montage, source_rate, config)
    expected_out = int(round(duration_seconds * config.sample_rate))
    data = data[:, :expected_out]
    if data.shape[-1] < expected_out:
        data = np.pad(data, ((0, 0), (0, expected_out - data.shape[-1])))
    return robust_scale(data, config.clip), mask


def quality_metrics(data: np.ndarray, sample_rate: float) -> dict[str, float]:
    """Simple window quality indicators for dashboard triage."""
    flat = np.mean(np.std(data, axis=-1) < 1e-4)
    clipped = np.mean(np.abs(data) >= 7.99)
    rms = float(np.sqrt(np.mean(np.square(data))))
    return {"flat_channel_fraction": float(flat), "clipped_fraction": float(clipped), "rms": rms}
