#Requires -Version 5.1
<#
.SYNOPSIS
  Push fresh AWS credentials to EC2 every N minutes while full-scale ingestion runs.

.DESCRIPTION
  BDSP access uses short-lived SSO/session tokens from your PC. Run this in a
  separate terminal alongside the pipeline so the EC2 downloader never stalls
  waiting for credential refresh.

  Prerequisites:
    1. aws login   (or otherwise valid local AWS session)
    2. SSM access to the instance

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File infra\watch-credentials.ps1 `
    -InstanceId i-02f81e8ff66b28592 -IntervalMinutes 5
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$InstanceId,
    [string]$Region = "us-east-1",
    [int]$IntervalMinutes = 5
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PushScript = Join-Path $ScriptDir "push-credentials-only.ps1"

Write-Host "Credential watchdog started. Pushing every $IntervalMinutes min to $InstanceId"
Write-Host "Keep this window open. Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    try {
        & powershell -ExecutionPolicy Bypass -File $PushScript -InstanceId $InstanceId -Region $Region
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$stamp] credentials pushed OK"
        } else {
            Write-Host "[$stamp] push failed (exit $LASTEXITCODE) - run 'aws login' locally"
        }
    } catch {
        Write-Host "[$stamp] push error: $_"
    }
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
