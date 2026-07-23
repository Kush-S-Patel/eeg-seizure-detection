# Neurotech full-scale EC2 provisioning (PowerShell)

Run from this repo on a machine with AWS CLI configured (`aws sts get-caller-identity`).

## BDSP credential note (important)

The EC2 instance role can read the deploy bucket, but **BDSP credentialed S3 access
is bound to your personal IAM credentials** (the same ones that work locally). The
instance profile alone will get `AccessDenied` on EDF downloads.

Before Phase B can run on EC2, configure AWS credentials on the instance:

```powershell
# Option A: copy your local profile (if you use ~/.aws/credentials)
.\infra\configure-remote-credentials.ps1 -InstanceId i-xxxxxxxx

# Option B: on the instance via SSM
aws ssm start-session --target i-xxxxxxxx --region us-east-1
sudo aws configure   # paste the same access key / secret that work locally
```

Then restart ingestion:

```bash
tmux kill-session -t fullscale 2>/dev/null || true
cd /opt/neurotech-seizure-detector/scripts
tmux new-session -d -s fullscale \
  "python downloader_100gb.py fullscale --skip-select --batch-gb 300 --cache-workers 14 --workers 16 2>&1 | tee /data/neurotech/fullscale.log"
```

Monitor: `tail -f /data/neurotech/fullscale.log`

## Quick start

```powershell
# 1. Provision EC2 + 2TB EBS (us-east-1, g5.4xlarge)
.\infra\provision.ps1

# 2. Copy repo to the instance (replace INSTANCE_ID)
.\infra\sync-to-instance.ps1 -InstanceId i-xxxxxxxx

# 3. Connect via SSM and finish setup
aws ssm start-session --target i-xxxxxxxx
# on the instance:
sudo bash /opt/neurotech-seizure-detector/infra/setup-on-instance.sh
```

## What gets created

| Resource | Purpose |
| --- | --- |
| IAM role + instance profile | S3 read on the BDSP access point, SSM access |
| Security group | Egress only (SSM shell, no open SSH required) |
| EC2 `g5.4xlarge` | 16 vCPU, 64GB RAM, A10G GPU, us-east-1 |
| EBS gp3 2048 GB | Rolling scratch + ~300GB window cache |

Estimated one-time ingestion: ~8-15 hours wall clock. Stop the instance when idle;
the EBS volume keeps billing (~$164/mo) until deleted.

## Full-scale pipeline on the instance

```bash
cd /opt/neurotech-seizure-detector
export NEUROTECH_DATA=/data/neurotech

# Planning stages (manifest + annotations can be copied from local machine)
python scripts/downloader_100gb.py manifest
python scripts/downloader_100gb.py metadata
python scripts/downloader_100gb.py annotations
python scripts/downloader_100gb.py index

# Full-scale ingestion (Phase A headers + Phase B rolling cache)
python scripts/downloader_100gb.py fullscale --batch-gb 300 --cache-workers 14

# Train on the complete cache
seizure-detector train --epochs 25 --output outputs/baseline
seizure-detector evaluate outputs/baseline/best.pt --split val
seizure-detector evaluate outputs/baseline/best.pt --split test
```

## Copying data from your local machine

You already have locally:

- `data/files.txt` (full S3 listing)
- `data/artifacts/manifest*.parquet`
- `data/raw/**/_Xltek.csv` (all 14,517 annotation CSVs)

Copy these to the instance to skip re-downloading annotations:

```powershell
.\infra\sync-to-instance.ps1 -InstanceId i-xxx -IncludeData
```

## Cost control

```powershell
# Stop compute (keeps EBS)
aws ec2 stop-instances --instance-ids i-xxxxxxxx

# Terminate instance (keeps EBS unless delete-on-termination is set)
aws ec2 terminate-instances --instance-ids i-xxxxxxxx

# Delete EBS when done
aws ec2 delete-volume --volume-id vol-xxxxxxxx
```
