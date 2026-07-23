#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
$Aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

$EnvLines = (& $Aws configure export-credentials --format env 2>&1)
if ($LASTEXITCODE -ne 0) {
    Write-Error "Could not export AWS credentials. Run 'aws login' locally first.`n$EnvLines"
    exit 1
}
$Vars = @{}
foreach ($line in $EnvLines) {
    if ($line -match '^export\s+(\w+)=(.+)$') { $Vars[$Matches[1]] = $Matches[2].Trim('"') }
}
if (-not $Vars.AWS_ACCESS_KEY_ID) {
    Write-Error "export-credentials returned no AWS_ACCESS_KEY_ID"
    exit 1
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
"@).Replace("`r", "")

$TempDir = Join-Path $env:TEMP "aws-creds-only"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$B64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Remote))
$Cmd = "echo '$B64' | base64 -d | bash"
$Params = @{ commands = @($Cmd) } | ConvertTo-Json
$ParamsFile = "$TempDir\ssm.json"
[System.IO.File]::WriteAllText($ParamsFile, $Params)

Write-Host "Pushing credentials only (no restart) to $InstanceId ..."
$CmdId = (& $Aws ssm send-command --region $Region --instance-ids $InstanceId --document-name AWS-RunShellScript --parameters "file://$ParamsFile" --query Command.CommandId --output text)
Start-Sleep -Seconds 15
$R = & $Aws ssm get-command-invocation --region $Region --command-id $CmdId --instance-id $InstanceId --output json | ConvertFrom-Json
Write-Host "Status: $($R.Status)"
Write-Host $R.StandardOutputContent
if ($R.StandardErrorContent) { Write-Host "ERR:" $R.StandardErrorContent }
