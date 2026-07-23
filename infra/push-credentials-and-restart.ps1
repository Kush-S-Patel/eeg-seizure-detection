#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
$Aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

$EnvLines = (& $Aws configure export-credentials --format env 2>&1)
$Vars = @{}
foreach ($line in $EnvLines) {
    if ($line -match '^export\s+(\w+)=(.+)$') { $Vars[$Matches[1]] = $Matches[2].Trim('"') }
}

$EnvFile = (@"
export AWS_ACCESS_KEY_ID=$($Vars.AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$($Vars.AWS_SECRET_ACCESS_KEY)
export AWS_SESSION_TOKEN=$($Vars.AWS_SESSION_TOKEN)
export AWS_EC2_METADATA_DISABLED=true
export AWS_DEFAULT_REGION=$Region
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export PATH=/opt/venv/bin:`$PATH
"@).Replace("`r", "")

$EnvB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($EnvFile.Trim()))

$Remote = (@"
#!/bin/bash
set -eu
echo '$EnvB64' | base64 -d > /home/ubuntu/.bdsp-aws-env
chmod 600 /home/ubuntu/.bdsp-aws-env
chown ubuntu:ubuntu /home/ubuntu/.bdsp-aws-env
. /home/ubuntu/.bdsp-aws-env
aws sts get-caller-identity
aws s3api head-object --bucket bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias --key EEG/bids/Neurotech/sub-Neurotech1/ses-1/eeg/sub-Neurotech1_ses-1_task-EEG_eeg.edf --region us-east-1 --query ContentLength
cat > /home/ubuntu/run-fullscale.sh <<'RUNEOF'
#!/bin/bash
set -eu
export PYTHONUNBUFFERED=1
export NEUROTECH_AWS_ENV_FILE=/home/ubuntu/.bdsp-aws-env
set -a
. /home/ubuntu/.bdsp-aws-env
set +a
cd /opt/neurotech-seizure-detector/scripts
exec python -u downloader_100gb.py fullscale --skip-select --skip-phase-a --batch-gb 300 --cache-workers 14 --workers 16
RUNEOF
chmod +x /home/ubuntu/run-fullscale.sh
chown ubuntu:ubuntu /home/ubuntu/run-fullscale.sh
chown -R ubuntu:ubuntu /data/neurotech
chmod -R u+rwX /data/neurotech
> /data/neurotech/fullscale.log
chown ubuntu:ubuntu /data/neurotech/fullscale.log
sudo -u ubuntu tmux kill-session -t fullscale 2>/dev/null || true
sudo -u ubuntu tmux new-session -d -s fullscale "bash /home/ubuntu/run-fullscale.sh 2>&1 | tee /data/neurotech/fullscale.log"
sleep 30
tail -n 15 /data/neurotech/fullscale.log
"@).Replace("`r", "")

$TempDir = Join-Path $env:TEMP "aws-creds-push"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
[System.IO.File]::WriteAllText("$TempDir\install.sh", $Remote.Replace("`r`n", "`n"))

$Params = @{ commands = @("bash /dev/stdin <<'EOF'`n$(Get-Content $TempDir\install.sh -Raw)`nEOF") } | ConvertTo-Json
# SSM JSON can't embed heredoc easily; upload script via base64 in one command instead
$B64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content "$TempDir\install.sh" -Raw)))
$Cmd = "echo '$B64' | base64 -d > /tmp/install.sh && bash /tmp/install.sh"
$Params = @{ commands = @($Cmd) } | ConvertTo-Json
$ParamsFile = "$TempDir\ssm.json"
[System.IO.File]::WriteAllText($ParamsFile, $Params)

Write-Host "Pushing credentials and restarting pipeline on $InstanceId ..."
$CmdId = (& $Aws ssm send-command --region $Region --instance-ids $InstanceId --document-name AWS-RunShellScript --parameters "file://$ParamsFile" --query Command.CommandId --output text)
Start-Sleep -Seconds 55
$R = & $Aws ssm get-command-invocation --region $Region --command-id $CmdId --instance-id $InstanceId --output json | ConvertFrom-Json
Write-Host "Status: $($R.Status)"
Write-Host $R.StandardOutputContent
if ($R.StandardErrorContent) { Write-Host "ERR:" $R.StandardErrorContent }
