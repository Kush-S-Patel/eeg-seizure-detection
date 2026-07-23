"""Tests for S3 cp skip logic (header stubs vs full EDFs)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pipeline import awscli


def test_cp_skips_when_size_matches(tmp_path: Path, monkeypatch):
    local = tmp_path / "rec_eeg.edf"
    local.write_bytes(b"x" * 1000)

    def fail_run(args):
        raise AssertionError("should not download when size matches")

    monkeypatch.setattr(awscli, "_run", fail_run)
    assert awscli.cp("rec_eeg.edf", local, expected_size_bytes=1000) is True


def test_cp_redownloads_header_stub(tmp_path: Path, monkeypatch):
    local = tmp_path / "rec_eeg.edf"
    local.write_bytes(b"x" * 256)
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        local.write_bytes(b"y" * 5000)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(awscli, "_run", fake_run)
    assert awscli.cp("rec_eeg.edf", local, expected_size_bytes=5000) is True
    assert len(calls) == 1
    assert local.stat().st_size == 5000


def test_apply_creds_file(tmp_path: Path, monkeypatch):
    creds = tmp_path / "creds.env"
    creds.write_text(
        "export AWS_ACCESS_KEY_ID=NEWKEY\nexport AWS_SECRET_ACCESS_KEY=secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEUROTECH_AWS_ENV_FILE", str(creds))
    env = awscli._apply_creds_file({"AWS_ACCESS_KEY_ID": "OLDKEY"})
    assert env["AWS_ACCESS_KEY_ID"] == "NEWKEY"


def test_is_auth_error():
    assert awscli._is_auth_error("ExpiredToken: token expired")
    assert awscli._is_auth_error(
        "fatal error: An error occurred (400) when calling the HeadObject operation: Bad Request"
    )
    assert not awscli._is_auth_error("An error occurred (404) when calling the HeadObject operation")


def test_wait_for_valid_creds(monkeypatch):
    calls = {"n": 0}

    def fake_valid():
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(awscli, "credentials_valid", fake_valid)
    monkeypatch.setattr(awscli.time, "sleep", lambda _: None)
    assert awscli.wait_for_valid_creds(poll_s=0, max_wait_s=60, log_interval_s=999) is True
