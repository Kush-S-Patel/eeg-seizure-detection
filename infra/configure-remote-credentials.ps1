#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
$Aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$CredsFile = "$env:USERPROFILE\.aws\credentials"
$ConfigFile = "$env:USERPROFILE\.aws\config"

if (-not (Test-Path $CredsFile)) { throw "No local credentials at $CredsFile" }

$CredsB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content $CredsFile -Raw)))
$ConfigB64 = if (Test-Path $ConfigFile) {
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content $ConfigFile -Raw)))
} else { "" }

$Cmd = @"
mkdir -p /root/.aws /home/ubuntu/.aws
echo '$CredsB64' | base64 -d > /root/.aws/credentials
echo '$CredsB64' | base64 -d > /home/ubuntu/.aws/credentials
if [ -n '$ConfigB64' ]; then
  echo '$ConfigB64' | base64 -d > /root/.aws/config
  echo '$ConfigB64' | base64 -d > /home/ubuntu/.aws/config
fi
chmod 600 /root/.aws/credentials /home/ubuntu/.aws/credentials
chown -R ubuntu:ubuntu /home/ubuntu/.aws
aws sts get-caller-identity --region $Region
"@

$Params = @{ commands = @($Cmd) } | ConvertTo-Json
$ParamsFile = Join-Path $env:TEMP "ssm-creds.json"
$Params | Set-Content $ParamsFile -Encoding ASCII

Write-Host "Copying local AWS credentials to instance (required for BDSP DUA access)..."
$CmdId = (& $Aws ssm send-command --region $Region --instance-ids $InstanceId `
    --document-name "AWS-RunShellScript" --parameters "file://$ParamsFile" `
    --query "Command.CommandId" --output text)
Start-Sleep -Seconds 15
& $Aws ssm get-command-invocation --region $Region --command-id $CmdId --instance-id $InstanceId `
    --query "[Status,StandardOutputContent,StandardErrorContent]" --output text

Write-Host "Restart fullscale: aws ssm send-command ... or re-run deploy-via-s3.ps1 -StartFullscale"
