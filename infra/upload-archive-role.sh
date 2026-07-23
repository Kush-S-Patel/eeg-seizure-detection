#!/bin/bash
set -eu
export PATH=/usr/bin:/bin:/usr/local/bin
mv /root/.aws/credentials /root/.aws/credentials.bak.expired 2>/dev/null || true
mv /root/.aws/config /root/.aws/config.bak 2>/dev/null || true
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
unset AWS_PROFILE AWS_EC2_METADATA_DISABLED AWS_SHARED_CREDENTIALS_FILE
export AWS_DEFAULT_REGION=us-east-1
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
LOG=/data/neurotech/s3-upload.log
echo "START $(date -u) instance-role" | tee "$LOG"
aws sts get-caller-identity | tee -a "$LOG"
aws s3 sync /data/neurotech/data/artifacts/ s3://neurotech-fullscale-904583676897/archive/artifacts/ --region us-east-1 2>&1 | tee -a "$LOG"
echo "DONE $(date -u) EXIT:${PIPESTATUS[0]}" | tee -a "$LOG"