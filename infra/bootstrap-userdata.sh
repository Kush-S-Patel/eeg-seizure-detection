#!/bin/bash
# Bootstrap script for Neurotech full-scale EC2 training instance.
# Installed via EC2 user-data on first boot.
set -euo pipefail

DATA_MOUNT="/data"
REPO_DIR="/opt/neurotech-seizure-detector"

# Wait for and mount the attached EBS volume (assumes /dev/nvme1n1 or /dev/xvdf).
for dev in /dev/nvme1n1 /dev/xvdf /dev/sdf; do
  if [ -b "$dev" ]; then
    if ! blkid "$dev" | grep -q ext4; then
      mkfs.ext4 -F "$dev"
    fi
    mkdir -p "$DATA_MOUNT"
    mount "$dev" "$DATA_MOUNT"
    grep -q "$dev" /etc/fstab || echo "$dev $DATA_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
    break
  fi
done

mkdir -p "$DATA_MOUNT/neurotech"/{data/raw,data/artifacts,outputs}
chown -R ubuntu:ubuntu "$DATA_MOUNT/neurotech" 2>/dev/null || true

# System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git awscli nvme-cli

# CUDA driver (Deep Learning AMI may already include this; harmless if present)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  apt-get install -y ubuntu-drivers-common || true
fi

# Python venv for the project (code copied separately via scp/rsync)
python3.11 -m venv /opt/venv
/opt/venv/bin/pip install --upgrade pip wheel

cat >/etc/profile.d/neurotech.sh <<'EOF'
export NEUROTECH_DATA="/data/neurotech"
export PATH="/opt/venv/bin:$PATH"
EOF

echo "Bootstrap complete. Mount: $DATA_MOUNT. Copy repo to $REPO_DIR and run infra/setup-on-instance.sh"
