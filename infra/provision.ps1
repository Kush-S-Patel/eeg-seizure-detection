#Requires -Version 5.1
param(
    [string]$Region = "us-east-1",
    [string]$InstanceType = "g5.4xlarge",
    [int]$VolumeGb = 2048,
    [string]$KeyName = "",
    [string]$AccessPoint = "bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
)

$ErrorActionPreference = "Stop"
$Aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $Aws)) { $Aws = "aws" }

$Project = "neurotech-fullscale"
$RoleName = "$Project-ec2-role"
$ProfileName = "$Project-ec2-profile"
$SgName = "$Project-sg"
$TempDir = Join-Path $env:TEMP "neurotech-provision"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

Write-Host "Resolving Ubuntu 22.04 AMI in $Region..."
$AmiId = & $Aws ec2 describe-images --region $Region --owners 099720109477 `
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" `
    --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text
if (-not $AmiId -or $AmiId -eq "None") { throw "Could not find Ubuntu 22.04 AMI" }
Write-Host "  AMI: $AmiId"

# IAM role
$TrustPath = Join-Path $TempDir "trust.json"
@'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
'@ | Set-Content $TrustPath -Encoding ASCII -NoNewline

$RoleExists = $false
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $Aws iam get-role --role-name $RoleName | Out-Null
$ErrorActionPreference = $prevEap
if ($LASTEXITCODE -eq 0) {
    Write-Host "Using existing IAM role $RoleName"
} else {
    & $Aws iam create-role --role-name $RoleName --assume-role-policy-document "file://$TrustPath" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to create IAM role $RoleName" }
    Write-Host "Created IAM role $RoleName"
    Start-Sleep -Seconds 5
}

$S3PolicyPath = Join-Path $TempDir "s3-policy.json"
@"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::$AccessPoint",
        "arn:aws:s3:::$AccessPoint/*"
      ]
    }
  ]
}
"@ | Set-Content $S3PolicyPath -Encoding ASCII
& $Aws iam put-role-policy --role-name $RoleName --policy-name "$Project-s3-read" --policy-document "file://$S3PolicyPath" | Out-Null
& $Aws iam attach-role-policy --role-name $RoleName --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" | Out-Null

$ErrorActionPreference = "SilentlyContinue"
& $Aws iam get-instance-profile --instance-profile-name $ProfileName | Out-Null
$ErrorActionPreference = $prevEap
if ($LASTEXITCODE -ne 0) {
    & $Aws iam create-instance-profile --instance-profile-name $ProfileName | Out-Null
    Start-Sleep -Seconds 5
}
$ErrorActionPreference = "SilentlyContinue"
& $Aws iam add-role-to-instance-profile --instance-profile-name $ProfileName --role-name $RoleName | Out-Null
$ErrorActionPreference = $prevEap

Start-Sleep -Seconds 10

# Security group
$VpcId = (& $Aws ec2 describe-vpcs --region $Region --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
$SgId = (& $Aws ec2 describe-security-groups --region $Region --filters "Name=group-name,Values=$SgName" --query "SecurityGroups[0].GroupId" --output text 2>$null)
if (-not $SgId -or $SgId -eq "None") {
    $SgId = (& $Aws ec2 create-security-group --region $Region --group-name $SgName --description "Neurotech fullscale EC2" --vpc-id $VpcId --query GroupId --output text)
    Write-Host "Created security group $SgId"
} else {
    Write-Host "Using security group $SgId"
}

# User data + block device mappings as files
$UserDataPath = Join-Path $PSScriptRoot "bootstrap-userdata.sh"
$UserDataB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Content $UserDataPath -Raw)))

$BdmPath = Join-Path $TempDir "block-device.json"
@'
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]
'@ | Set-Content $BdmPath -Encoding ASCII -NoNewline

$LaunchArgs = @(
    "ec2", "run-instances",
    "--region", $Region,
    "--image-id", $AmiId,
    "--instance-type", $InstanceType,
    "--iam-instance-profile", "Name=$ProfileName",
    "--security-group-ids", $SgId,
    "--block-device-mappings", "file://$BdmPath",
    "--user-data", $UserDataB64,
    "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=$Project}]",
    "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
    "--output", "json"
)
if ($KeyName) { $LaunchArgs += @("--key-name", $KeyName) }

Write-Host "Launching $InstanceType..."
$LaunchJson = & $Aws @LaunchArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    if ($InstanceType -like "g5.*") {
        Write-Host "GPU instance limit hit; falling back to c6i.4xlarge for ingestion."
        $InstanceType = "c6i.4xlarge"
        $LaunchArgs = $LaunchArgs | ForEach-Object { if ($_ -eq "g5.4xlarge" -or $_ -like "g5.*") { "c6i.4xlarge" } else { $_ } }
        # Replace instance type arg
        for ($i = 0; $i -lt $LaunchArgs.Count; $i++) {
            if ($LaunchArgs[$i] -eq "--instance-type") { $LaunchArgs[$i + 1] = "c6i.4xlarge"; break }
        }
        $LaunchJson = & $Aws @LaunchArgs 2>&1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host $LaunchJson
        throw "Failed to launch instance: $LaunchJson"
    }
}
$Launch = $LaunchJson | ConvertFrom-Json
$InstanceId = $Launch.Instances[0].InstanceId
$Az = $Launch.Instances[0].Placement.AvailabilityZone
Write-Host "  InstanceId: $InstanceId ($Az)"

Write-Host "Creating ${VolumeGb}GB gp3 data volume..."
$VolJson = & $Aws ec2 create-volume --region $Region --availability-zone $Az --size $VolumeGb --volume-type gp3 `
    --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=$Project-data}]" --output json
$VolumeId = ($VolJson | ConvertFrom-Json).VolumeId
Write-Host "  VolumeId: $VolumeId"
& $Aws ec2 wait volume-available --region $Region --volume-ids $VolumeId
& $Aws ec2 wait instance-running --region $Region --instance-ids $InstanceId
& $Aws ec2 attach-volume --region $Region --volume-id $VolumeId --instance-id $InstanceId --device /dev/sdf | Out-Null

$StateFile = Join-Path $PSScriptRoot "instance-state.json"
@{
    region = $Region
    instanceId = $InstanceId
    volumeId = $VolumeId
    instanceType = $InstanceType
    amiId = $AmiId
    createdAt = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content $StateFile

Write-Host ""
Write-Host "Provisioned successfully."
Write-Host "  Instance: $InstanceId"
Write-Host "  Volume:   $VolumeId"
Write-Host "  State:    $StateFile"
Write-Host ""
Write-Host "Connect: aws ssm start-session --target $InstanceId --region $Region"
Write-Host "Transfer: .\infra\sync-to-instance.ps1 -InstanceId $InstanceId"
