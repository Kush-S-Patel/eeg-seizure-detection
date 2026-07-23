#!/bin/bash
# Start full-scale ingestion in tmux (survives SSM disconnect).
set -eu
export PATH=/opt/venv/bin:$PATH
export NEUROTECH_DATA=/data/neurotech
cd /opt/neurotech-seizure-detector/scripts

apt-get install -y tmux 2>/dev/null || true

tmux kill-session -t fullscale 2>/dev/null || true
tmux new-session -d -s fullscale \
  "AWS_EC2_METADATA_DISABLED=true AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required PATH=/opt/venv/bin:\$PATH python downloader_100gb.py fullscale --skip-select --batch-gb 300 --cache-workers 14 --workers 16 2>&1 | tee /data/neurotech/fullscale.log"

echo "Started fullscale in tmux session 'fullscale'"
echo "Monitor: tmux attach -t fullscale  OR  tail -f /data/neurotech/fullscale.log"
