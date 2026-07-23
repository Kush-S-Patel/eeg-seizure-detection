"""Tests for EDF header-only parsing."""

from __future__ import annotations

from pathlib import Path

from seizure_detector.preprocess import parse_edf_header


def _write_minimal_edf(path: Path, *, num_records: int = 100, record_duration: float = 1.0) -> None:
    num_signals = 2
    header_bytes = 256 + num_signals * 256
    main = bytearray(256)
    main[184:192] = f"{header_bytes:<8}".encode("ascii")
    main[200:208] = f"{num_records:<8}".encode("ascii")
    main[208:216] = f"{record_duration:<8}".encode("ascii")
    main[216:224] = f"{num_signals:<8}".encode("ascii")
    main[168:176] = b"01.01.23"
    main[176:184] = b"00.00.00"

    blocks = bytearray()
    for label, samples in (("FP1", 256), ("FP2", 256)):
        block = bytearray(256)
        block[0:16] = f"{label:<16}".encode("ascii")
        block[216:224] = f"{samples:<8}".encode("ascii")
        blocks.extend(block)

    path.write_bytes(bytes(main) + bytes(blocks))


def test_parse_edf_header_stub(tmp_path: Path):
    edf = tmp_path / "stub.edf"
    _write_minimal_edf(edf, num_records=120, record_duration=1.0)
    info = parse_edf_header(edf)
    assert info["duration_seconds"] == 120.0
    assert info["sample_rate"] == 256.0
    assert info["channels"] == ["FP1", "FP2"]
