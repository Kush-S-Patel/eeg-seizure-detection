#!/bin/bash
# Remote setup script - uploaded to S3 and executed via SSM.
set -eu
export DEBIAN_FRONTEND=noninteractive
REGION="${AWS_REGION:-us-east-1}"
BUCKET="${DEPLOY_BUCKET:?set DEPLOY_BUCKET}"

apt-get update -y
apt-get install -y unzip python3.11 python3.11-venv awscli nvme-cli

# Mount data volume if present
for dev in /dev/nvme1n1 /dev/xvdf /dev/sdf; do
  if [ -b "$dev" ]; then
    if ! blkid "$dev" | grep -q ext4; then
      mkfs.ext4 -F "$dev"
    fi
    mkdir -p /data
    mount "$dev" /data || true
    grep -q "$dev" /etc/fstab || echo "$dev /data ext4 defaults,nofail 0 2" >> /etc/fstab
    break
  fi
done

mkdir -p /opt/neurotech-seizure-detector /data/neurotech/data
aws s3 cp "s3://${BUCKET}/deploy/neurotech-seizure-detector.zip" /tmp/repo.zip --region "$REGION"
unzip -o /tmp/repo.zip -d /opt/neurotech-seizure-detector

if aws s3 ls "s3://${BUCKET}/data/" --region "$REGION" 2>/dev/null; then
  aws s3 sync "s3://${BUCKET}/data/" /data/neurotech/data/ --region "$REGION"
fi

python3.11 -m venv /opt/venv
/opt/venv/bin/pip install --upgrade pip wheel
cd /opt/neurotech-seizure-detector
/opt/venv/bin/pip install -e ".[dev]"
ln -sf /data/neurotech/data /opt/neurotech-seizure-detector/data
chown -R ubuntu:ubuntu /data/neurotech /opt/neurotech-seizure-detector
chmod -R u+rwX /data/neurotech

cat >> /home/ubuntu/.bashrc <<'EOF'
export PATH=/opt/venv/bin:$PATH
export NEUROTECH_DATA=/data/neurotech
EOF

echo "Setup complete on $(hostname)"
