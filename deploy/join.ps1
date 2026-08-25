<#
  MeshPool — join this machine to a pool (run on each laptop you want to contribute).

  Usage (from the meshpool folder), using the command shown on the control-plane machine:
    powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://192.168.1.20:8080 -Token abc123
    powershell -ExecutionPolicy Bypass -File deploy\join.ps1 -Server http://192.168.1.20:8080 -Token abc123 -Cpu 40 -OpenFirewall

  The agent auto-detects THIS machine's LAN IP and self-registers. Leave it running.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string]$Server,
  [string]$Token = "",
  [string]$Name = $env:COMPUTERNAME,
  [ValidateRange(0, 100)] [int]$Cpu = 50,
  [int]$Port = 9001,
  [switch]$OpenFirewall
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$Py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $Py) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Py) { throw "Python 3.11+ not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and re-run." }

Push-Location $RepoDir
try {
  Write-Host "==> Installing dependencies (first run only)..."
  & $Py -m pip install --quiet --disable-pip-version-check -r requirements.txt

  # Preflight: can we reach the control plane? Fail fast with a clear message.
  try {
    $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 -Uri "$Server/api/v1/health"
    Write-Host "==> Control plane reachable at $Server"
  } catch {
    throw "Cannot reach control plane at $Server. Check the address, that the control-plane machine is running, and that its firewall allows the port."
  }

  if ($OpenFirewall) {
    try {
      netsh advfirewall firewall delete rule name="MeshPool Agent" 2>$null | Out-Null
      netsh advfirewall firewall add rule name="MeshPool Agent" dir=in action=allow protocol=TCP localport=$Port | Out-Null
      Write-Host "==> Firewall: allowed inbound TCP $Port"
    } catch {
      Write-Warning "Could not add firewall rule (run PowerShell as Administrator, or allow the prompt). Port $Port must be reachable from the control-plane machine."
    }
  }

  $env:MESHPOOL_CONTROL_PLANE_URL = $Server
  $env:MESHPOOL_ENROLL_TOKEN = $Token
  $env:MESHPOOL_NAME = $Name
  $env:MESHPOOL_DONATE_CPU_PCT = "$Cpu"
  $env:MESHPOOL_AGENT_PORT = "$Port"
  # Advertise host is auto-detected (LAN IP). Bind on all interfaces.
  Remove-Item Env:\MESHPOOL_AGENT_HOST -ErrorAction SilentlyContinue

  # Friendly Ollama preflight (non-fatal): this machine can serve any Ollama model it
  # holds to the whole pool. Tell the user what it will contribute.
  try {
    $tags = Invoke-RestMethod -TimeoutSec 2 -Uri "http://127.0.0.1:11434/api/tags"
    $models = @($tags.models | ForEach-Object { $_.name })
    if ($models.Count -gt 0) {
      Write-Host "==> Ollama detected here: $($models -join ', ')  (these become available to the pool)"
    } else {
      Write-Host "==> Ollama is running but has no models. Pull one so this machine can serve it:  ollama pull qwen2.5-coder:7b"
    }
  } catch {
    Write-Host "==> No local Ollama (optional). To have this machine serve a real LLM, install Ollama"
    Write-Host "    (https://ollama.com/download) and run 'ollama pull qwen2.5-coder:7b', then re-run this join."
  }

  Write-Host "==> Joining $Server as '$Name' (donating $Cpu% CPU)"
  Write-Host "    If Windows asks about network access, choose 'Allow' on Private networks."
  & $Py -m meshpool.agent.daemon
}
finally {
  Pop-Location
}
