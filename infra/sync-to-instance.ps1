#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [switch]$IncludeData,
    [string]$RemoteDir = "/opt/neurotech-seizure-detector",
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $Aws)) { $Aws = "aws" }

# Resolve public IP for scp (fallback if SSM port forwarding not used)
$PublicIp = (& $Aws ec2 describe-instances --region $Region --instance-ids $InstanceId `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

if (-not $PublicIp -or $PublicIp -eq "None") {
    Write-Host "No public IP on instance. Use SSM or assign an Elastic IP."
    Write-Host "Alternative: zip repo and upload via S3, then pull on instance."
    Write-Host ""
    Write-Host "Creating repo archive for manual transfer..."
    $ZipPath = Join-Path $env:TEMP "neurotech-seizure-detector.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path "$RepoRoot\*" -DestinationPath $ZipPath -Force `
        -CompressionLevel Optimal
    Write-Host "  Archive: $ZipPath"
    Write-Host "  Upload to S3 and extract on instance, or enable a public IP + key pair for scp."
    exit 0
}

Write-Host "Syncing repo to ubuntu@${PublicIp}:$RemoteDir ..."
Write-Host "(Requires SSH key configured for the instance.)"
scp -r "$RepoRoot\src" "$RepoRoot\scripts" "$RepoRoot\infra" "$RepoRoot\pyproject.toml" `
    "$RepoRoot\requirements.txt" "$RepoRoot\app.py" `
    "ubuntu@${PublicIp}:$RemoteDir/"

if ($IncludeData) {
    Write-Host "Syncing local data artifacts and annotations..."
    scp -r "$RepoRoot\data\files.txt" "$RepoRoot\data\artifacts" `
        "ubuntu@${PublicIp}:/data/neurotech/data/"
    scp -r "$RepoRoot\data\raw\EEG" "ubuntu@${PublicIp}:/data/neurotech/data/raw/"
}

Write-Host "Done. SSH: ssh ubuntu@$PublicIp"
