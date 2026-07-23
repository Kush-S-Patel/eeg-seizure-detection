#!/usr/bin/env python3
"""Push patched source files to EC2 via SSM (one file per command to stay under 97KB)."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "i-02f81e8ff66b28592"
REGION = "us-east-1"
AWS = r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"

FILES = [
    "src/seizure_detector/preprocess.py",
    "src/seizure_detector/windows.py",
    "src/seizure_detector/cache.py",
    "scripts/pipeline/awscli.py",
    "scripts/pipeline/download.py",
    "scripts/pipeline/fullscale.py",
    "scripts/pipeline/config.py",
]


def _send(commands: list[str]) -> tuple[str, dict]:
    params_file = Path(tempfile.gettempdir()) / "ssm-codepush.json"
    params_file.write_text(json.dumps({"commands": commands}), encoding="ascii")
    cmd_id = subprocess.check_output(
        [
            AWS,
            "ssm",
            "send-command",
            "--region",
            REGION,
            "--instance-ids",
            INSTANCE,
            "--document-name",
            "AWS-RunShellScript",
            "--parameters",
            f"file://{params_file}",
            "--query",
            "Command.CommandId",
            "--output",
            "text",
        ],
        text=True,
    ).strip()
    time.sleep(8)
    result = subprocess.check_output(
        [
            AWS,
            "ssm",
            "get-command-invocation",
            "--region",
            REGION,
            "--command-id",
            cmd_id,
            "--instance-id",
            INSTANCE,
            "--output",
            "json",
        ],
        text=True,
    )
    return cmd_id, json.loads(result)


def main() -> int:
    ok = True
    for rel in FILES:
        data = (ROOT / rel).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        dest = f"/opt/neurotech-seizure-detector/{rel}"
        cmd_id, payload = _send([f"echo '{b64}' | base64 -d > {dest}"])
        status = payload["Status"]
        print(f"{rel}: {status} ({cmd_id})")
        if status != "Success":
            ok = False
            err = payload.get("StandardErrorContent", "")
            if err:
                print("  ERR:", err[:200])

    _, payload = _send(
        [
            "chown -R ubuntu:ubuntu /opt/neurotech-seizure-detector/src "
            "/opt/neurotech-seizure-detector/scripts",
            "grep -c wait_for_valid_creds "
            "/opt/neurotech-seizure-detector/scripts/pipeline/awscli.py",
        ]
    )
    print("verify:", payload.get("StandardOutputContent", "").strip())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
