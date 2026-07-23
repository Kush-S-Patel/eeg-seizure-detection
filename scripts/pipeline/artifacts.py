"""Load/save helpers for pipeline artifacts.

Prefers Parquet (compact, typed) but transparently falls back to CSV when
``pyarrow``/``fastparquet`` is unavailable, so the pipeline runs on a bare
pandas install.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _csv_twin(path: Path) -> Path:
    return path.with_suffix(".csv")


def save_df(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path``; returns the path actually written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        written = path
    except Exception:
        written = _csv_twin(path)
        df.to_csv(written, index=False)
    print(f"  wrote {written}  ({len(df):,} rows)")
    return written


def load_df(path: Path) -> pd.DataFrame:
    """Load an artifact, trying Parquet then the CSV twin."""
    path = Path(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv = _csv_twin(path)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Neither {path} nor {csv} exists. Run the stage that produces it first."
    )


def exists(path: Path) -> bool:
    path = Path(path)
    return path.exists() or _csv_twin(path).exists()
