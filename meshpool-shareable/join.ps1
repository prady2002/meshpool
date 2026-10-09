<#
  MeshPool - Join an existing cluster as a worker node (Windows / PowerShell).
  Usage:
    powershell -ExecutionPolicy Bypass -File join.ps1 -Server http://192.168.1.20:8080 -Token <TOKEN> -Cpu 50
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string]$Server,
  [string]$Token = "",
  [ValidateRange(0, 100)] [int]$Cpu = 50,
  [string]$Name = "",
  [int]$AgentPort = 9001,
  [string]$AdvertiseHost = ""
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

$Py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $Py) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Py) {
  Write-Host "Python 3.11+ is required. Install from https://www.python.org/downloads/"
  exit 1
}

Push-Location $Root
try {
  $marker = Join-Path $Root ".installed-deps"
  $whlItem = Get-ChildItem -Path $Root -Filter "meshpool-*.whl" | Select-Object -First 1
  if (-not (Test-Path $marker)) {
    Write-Host "==> First run: Installing MeshPool build package..."
    if ($whlItem) {
      & $Py -m pip install --quiet --disable-pip-version-check --no-warn-script-location $whlItem.FullName
    }
    if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Path $marker -Force | Out-Null }
  }

  if (-not $AdvertiseHost) {
    try {
      $AdvertiseHost = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.PrefixOrigin -ne "WellKnown" } |
        Select-Object -First 1).IPAddress
    } catch { }
    if (-not $AdvertiseHost) { $AdvertiseHost = $env:COMPUTERNAME }
  }

  $env:MESHPOOL_CONTROL_PLANE_URL = $Server
  if ($Token) { $env:MESHPOOL_INVITE_TOKEN = $Token }
  $env:MESHPOOL_DONATE_CPU_PCT = [string]$Cpu
  $nodeName = if ($Name) { $Name } else { $env:COMPUTERNAME }
  $env:MESHPOOL_NAME = $nodeName
  $env:MESHPOOL_AGENT_HOST = $AdvertiseHost
  $env:MESHPOOL_AGENT_PORT = [string]$AgentPort

  Write-Host "==> Joining MeshPool at $Server as node '$nodeName' ($AdvertiseHost)..."
  & $Py -m meshpool.agent.daemon
}
finally { Pop-Location }
