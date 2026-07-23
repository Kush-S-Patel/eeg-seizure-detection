"""Small retrying wrappers around the AWS CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from . import config


def s3_uri(key: str) -> str:
    return f"s3://{config.ACCESS_POINT}/{key.lstrip('/')}"


def _creds_file() -> Path:
    return Path(os.environ.get("NEUROTECH_AWS_ENV_FILE", "/home/ubuntu/.bdsp-aws-env"))


def _apply_creds_file(env: dict[str, str]) -> dict[str, str]:
    """Merge credentials from the on-disk env file (refreshed via SSM push)."""
    path = _creds_file()
    if not path.is_file():
        return env
    aws_keys = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_REQUEST_CHECKSUM_CALCULATION",
        "AWS_RESPONSE_CHECKSUM_VALIDATION",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        if key not in aws_keys:
            continue
        env[key] = value.strip().strip('"').strip("'")
    return env


def _aws_executable() -> str:
    """Resolve AWS CLI path even if PATH was corrupted."""
    cli = config.AWS_CLI
    if Path(cli).is_file():
        return cli
    found = shutil.which(cli)
    if found:
        return found
    for fallback in ("/usr/bin/aws", "/usr/local/bin/aws"):
        if Path(fallback).is_file():
            return fallback
    return cli


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = _apply_creds_file(os.environ.copy())
    # BDSP objects predate newer automatic checksum behavior. Restricting
    # validation to modeled requirements avoids AWS CLI transfer regressions.
    env.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    env.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
    # Prefer explicit user creds over the instance profile when both exist.
    if env.get("AWS_ACCESS_KEY_ID") or env.get("AWS_PROFILE"):
        env.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    return subprocess.run(
        [_aws_executable(), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _is_missing(stderr: str) -> bool:
    value = stderr.lower()
    return any(
        marker in value
        for marker in ("(404)", "not found", "does not exist", "nosuchkey")
    )


def _is_auth_error(stderr: str) -> bool:
    value = stderr.lower()
    if any(
        marker in value
        for marker in (
            "expiredtoken",
            "expired token",
            "invalidtoken",
            "security token included in the request is expired",
            "security token included in the request is invalid",
        )
    ):
        return True
    return "headobject" in value and "400" in value and "bad request" in value


def credentials_valid() -> bool:
    """True when the on-disk creds file yields a working STS identity."""
    result = _run(["sts", "get-caller-identity"])
    return result.returncode == 0


def wait_for_valid_creds(
    *,
    poll_s: float = 15.0,
    max_wait_s: float | None = None,
    log_interval_s: float = 60.0,
) -> bool:
    """Block until credentials work again (pushed from the Windows watchdog).

    Returns False only when ``max_wait_s`` elapses without valid credentials.
    """
    if max_wait_s is None:
        max_wait_s = float(os.environ.get("NEUROTECH_AUTH_WAIT_S", "86400"))
    deadline = time.time() + max_wait_s
    last_log = 0.0
    while time.time() < deadline:
        if credentials_valid():
            return True
        now = time.time()
        if now - last_log >= log_interval_s:
            print(
                f"    waiting for fresh AWS credentials in {_creds_file()} "
                f"(run infra/watch-credentials.ps1 on your PC)...",
                flush=True,
            )
            last_log = now
        time.sleep(poll_s)
    return False


def cp(
    key: str,
    local_path: Path,
    *,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    quiet: bool = True,
    warn_on_missing: bool = True,
    expected_size_bytes: int | None = None,
) -> bool:
    """Copy one object, skipping complete local files and permanent 404s.

    When ``expected_size_bytes`` is set (typical for EDF signal files), a local
    file is treated as complete only if its size matches. Phase A leaves tiny
    header stubs that must not block Phase B full downloads.
    """
    local_path = Path(local_path)
    if local_path.exists():
        local_size = local_path.stat().st_size
        if expected_size_bytes is not None:
            if local_size == expected_size_bytes:
                return True
            if local_size > 0:
                local_path.unlink(missing_ok=True)
        elif local_size > 0:
            return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["s3", "cp", s3_uri(key), str(local_path)]
    if quiet:
        args.append("--only-show-errors")

    last_error = ""
    transient_attempts = 0
    while True:
        result = _run(args)
        if result.returncode == 0:
            return True
        last_error = (result.stderr or result.stdout or "").strip()
        if _is_missing(last_error):
            if warn_on_missing:
                print(f"    - missing (skipped): {key}", flush=True)
            return False
        if _is_auth_error(last_error):
            print(f"    ! auth error, waiting for credential refresh: {key}", flush=True)
            if wait_for_valid_creds():
                continue
            print(f"    ! timed out waiting for credentials: {key}", flush=True)
            return False
        transient_attempts += 1
        if transient_attempts >= max_retries:
            break
        time.sleep(backoff_s**transient_attempts)

    print(f"    ! failed after {max_retries} tries: {key}\n      {last_error}", flush=True)
    return False


def get_range(
    key: str,
    local_path: Path,
    end_byte: int = 131071,
    *,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> bool:
    """Fetch only the first ``end_byte + 1`` bytes of an object (EDF header).

    Used in Phase A of full-scale ingestion: EDF headers are a few KB and
    contain duration/channel metadata, so we can build the complete window
    manifest without downloading multi-GB signal files.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "s3api",
        "get-object",
        "--bucket",
        config.ACCESS_POINT,
        "--key",
        key.lstrip("/"),
        "--range",
        f"bytes=0-{end_byte}",
        str(local_path),
    ]
    last_error = ""
    for attempt in range(1, max_retries + 1):
        result = _run(args)
        if result.returncode == 0 and local_path.exists() and local_path.stat().st_size > 0:
            return True
        last_error = (result.stderr or result.stdout or "").strip()
        if _is_missing(last_error):
            return False
        if attempt < max_retries:
            time.sleep(backoff_s**attempt)
    print(f"    ! range-get failed after {max_retries} tries: {key}\n      {last_error}")
    return False


def get_edf_header(
    key: str,
    local_path: Path,
    *,
    file_size_bytes: int | None = None,
    max_bytes: int = 262143,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> bool:
    """Fetch enough bytes to cover the full EDF header (two-step range GET)."""
    local_path = Path(local_path)
    if local_path.exists():
        try:
            from seizure_detector.preprocess import parse_edf_header

            parse_edf_header(local_path, file_size_bytes=file_size_bytes)
            return True
        except ValueError:
            local_path.unlink(missing_ok=True)

    probe = local_path.with_suffix(local_path.suffix + ".probe")
    if not get_range(key, probe, end_byte=255, max_retries=max_retries, backoff_s=backoff_s):
        probe.unlink(missing_ok=True)
        return False
    try:
        header_bytes = int(probe.read_bytes()[184:192].decode("ascii", errors="replace").strip())
    except ValueError:
        probe.unlink(missing_ok=True)
        return False
    probe.unlink(missing_ok=True)
    end_byte = min(max(header_bytes, 256) - 1, max_bytes)
    if not get_range(
        key,
        local_path,
        end_byte=end_byte,
        max_retries=max_retries,
        backoff_s=backoff_s,
    ):
        return False
    try:
        from seizure_detector.preprocess import parse_edf_header

        parse_edf_header(local_path, file_size_bytes=file_size_bytes)
        return True
    except ValueError as exc:
        print(f"    ! invalid EDF header for {key}: {exc}")
        local_path.unlink(missing_ok=True)
        return False


def sync(
    remote_prefix: str,
    local_dir: Path,
    *,
    include: str | None = None,
    exclude_all_first: bool = False,
    quiet: bool = True,
) -> bool:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    args = ["s3", "sync", s3_uri(remote_prefix), str(local_dir)]
    if exclude_all_first:
        args += ["--exclude", "*"]
    if include:
        args += ["--include", include]
    if quiet:
        args.append("--only-show-errors")
    result = _run(args)
    if result.returncode != 0:
        print((result.stderr or result.stdout or "").strip())
        return False
    return True
