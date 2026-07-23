#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [string]$Region = "us-east-1",
    [string]$Bucket = "",
    [switch]$IncludeData,
    [switch]$StartFullscale
)

$ErrorActionPreference = "Stop"
$Aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $Aws)) { $Aws = "aws" }
$RepoRoot = Split-Path $PSScriptRoot -Parent

if (-not $Bucket) {
    $Account = (& $Aws sts get-caller-identity --query Account --output text)
    $Bucket = "neurotech-fullscale-$Account"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $Aws s3 mb "s3://$Bucket" --region $Region | Out-Null
    $ErrorActionPreference = $prev
}

# Upload repo zip
$ZipPath = Join-Path $env:TEMP "neurotech-seizure-detector.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
$Items = @(
    "$RepoRoot\src", "$RepoRoot\scripts", "$RepoRoot\infra", "$RepoRoot\tests",
    "$RepoRoot\pyproject.toml", "$RepoRoot\requirements.txt", "$RepoRoot\app.py", "$RepoRoot\README.md"
)
Compress-Archive -Path $Items -DestinationPath $ZipPath -Force
Write-Host "Uploading repo zip..."
& $Aws s3 cp $ZipPath "s3://$Bucket/deploy/neurotech-seizure-detector.zip" --region $Region

# Upload remote scripts
& $Aws s3 cp "$PSScriptRoot\setup-remote.sh" "s3://$Bucket/deploy/setup-remote.sh" --region $Region
& $Aws s3 cp "$PSScriptRoot\start-fullscale-remote.sh" "s3://$Bucket/deploy/start-fullscale-remote.sh" --region $Region

if ($IncludeData) {
    Write-Host "Syncing artifacts (excluding window_cache)..."
    & $Aws s3 sync "$RepoRoot\data\artifacts" "s3://$Bucket/data/artifacts" --region $Region `
        --exclude "window_cache/*"
    if (Test-Path "$RepoRoot\data\files.txt") {
        & $Aws s3 cp "$RepoRoot\data\files.txt" "s3://$Bucket/data/files.txt" --region $Region
    }
    Write-Host "Syncing Xltek annotation CSVs..."
    & $Aws s3 sync "$RepoRoot\data\raw\EEG" "s3://$Bucket/data/raw/EEG" --region $Region `
        --exclude "*" --include "*_Xltek.csv"
}

function Invoke-SsmScript {
    param([string]$S3ScriptKey, [string]$Label)
    $Cmd = "export DEPLOY_BUCKET=$Bucket AWS_REGION=$Region; aws s3 cp s3://$Bucket/$S3ScriptKey /tmp/run.sh --region $Region && sed -i 's/\r$//' /tmp/run.sh && chmod +x /tmp/run.sh && bash /tmp/run.sh"
    $ParamsFile = Join-Path $env:TEMP "ssm-params.json"
    @{ commands = @($Cmd) } | ConvertTo-Json | Set-Content $ParamsFile -Encoding ASCII
    Write-Host "SSM: $Label ..."
    $CmdId = (& $Aws ssm send-command --region $Region --instance-ids $InstanceId `
        --document-name "AWS-RunShellScript" --parameters "file://$ParamsFile" `
        --query "Command.CommandId" --output text)
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 15
        $Status = (& $Aws ssm get-command-invocation --region $Region --command-id $CmdId `
            --instance-id $InstanceId --query Status --output text 2>$null)
        Write-Host "  $Label status: $Status"
        if ($Status -in @("Success", "Failed", "Cancelled", "TimedOut")) {
            $Out = (& $Aws ssm get-command-invocation --region $Region --command-id $CmdId `
                --instance-id $InstanceId --query StandardOutputContent --output text 2>$null)
            if ($Out) { Write-Host $Out }
            if ($Status -ne "Success") {
                $Err = (& $Aws ssm get-command-invocation --region $Region --command-id $CmdId `
                    --instance-id $InstanceId --query StandardErrorContent --output text 2>$null)
                if ($Err) { Write-Host $Err }
            }
            return $Status
        }
    }
    return "TimedOut"
}

Invoke-SsmScript "deploy/setup-remote.sh" "setup" | Out-Null

if ($StartFullscale) {
    Invoke-SsmScript "deploy/start-fullscale-remote.sh" "fullscale" | Out-Null
}

Write-Host ""
Write-Host "Done. Connect: aws ssm start-session --target $InstanceId --region $Region"
Write-Host "Log: tail -f /data/neurotech/fullscale.log"
