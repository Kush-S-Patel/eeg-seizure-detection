#!/bin/bash
# Run on the EC2 instance after code is copied to /opt/neurotech-seizure-detector
set -euo pipefail

REPO="/opt/neurotech-seizure-detector"
DATA="${NEUROTECH_DATA:-/data/neurotech}"
VENV="/opt/venv"

sudo mkdir -p "$REPO" "$DATA"/{data/raw,data/artifacts,outputs}
sudo chown -R "$USER:$USER" "$REPO" "$DATA" 2>/dev/null || true

cd "$REPO"
"$VENV/bin/pip" install -e ".[dev]"

# Symlink data dir if repo lives outside the volume
if [ ! -e "$REPO/data" ]; then
  ln -sf "$DATA/data" "$REPO/data"
fi

echo "Setup complete."
echo "  Repo: $REPO"
echo "  Data: $DATA"
echo ""
echo "Run full-scale ingestion:"
echo "  cd $REPO && python scripts/downloader_100gb.py fullscale --batch-gb 300 --cache-workers 14"
