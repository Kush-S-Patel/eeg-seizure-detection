from datetime import datetime

import pandas as pd

from seizure_detector.labels import is_seizure_text, read_seizure_times


def test_seizure_text_negation():
    assert is_seizure_text("@Clip: probable electrographic seizure")
    assert is_seizure_text("ictal onset")
    assert not is_seizure_text("no seizure activity")
    assert not is_seizure_text("patient is seizure-free")


def test_annotation_timestamp_alignment(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "Text": ["@Seizure", "No seizure", "@Spike"],
            "CreationTime": [
                "2023-01-01T00:01:00",
                "2023-01-01T00:02:00",
                "2023-01-01T00:03:00",
            ],
        }
    ).to_csv(path, index=False)
    offsets = read_seizure_times(path, datetime(2023, 1, 1))
    assert offsets.tolist() == [60.0]
