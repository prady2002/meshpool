<#
  MeshPool — start the control plane (run this on ONE machine on your LAN).

  Usage (from the meshpool folder):
    powershell -ExecutionPolicy Bypass -File deploy\start-controlplane.ps1
    powershell -ExecutionPolicy Bypass -File deploy\start-controlplane.ps1 -OpenFirewall

  This machine hosts the dashboard and coordinates the pool. Leave it running.
  Other laptops join with deploy\join.ps1 (the exact command is printed on start).
#>
[CmdletBinding()]
param(
  [int]$Port = 8080,
  [ValidateSet("home", "office", "enterprise")] [string]$Scale = "home",
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

  if ($OpenFirewall) {
    try {
      netsh advfirewall firewall delete rule name="MeshPool ControlPlane" 2>$null | Out-Null
      netsh advfirewall firewall add rule name="MeshPool ControlPlane" dir=in action=allow protocol=TCP localport=$Port | Out-Null
      Write-Host "==> Firewall: allowed inbound TCP $Port"
    } catch {
      Write-Warning "Could not add firewall rule (run PowerShell as Administrator, or allow the prompt). Port $Port must be reachable from other machines."
    }
  }

  $env:MESHPOOL_PORT = "$Port"
  $env:MESHPOOL_SCALE = $Scale

  # Friendly Ollama preflight (non-fatal). MeshPool can route real LLMs to any node that
  # runs them under Ollama - show what THIS machine will contribute to the pool.
  try {
    $tags = Invoke-RestMethod -TimeoutSec 2 -Uri "http://127.0.0.1:11434/api/tags"
    $models = @($tags.models | ForEach-Object { $_.name })
    if ($models.Count -gt 0) {
      Write-Host "==> Ollama detected here: $($models -join ', ')"
    } else {
      Write-Host "==> Ollama is running but has no models. To serve a real LLM across the pool, run:  ollama pull qwen2.5-coder:7b"
    }
  } catch {
    Write-Host "==> No local Ollama (optional). For real LLMs across the pool, install Ollama"
    Write-Host "    (https://ollama.com/download) and run 'ollama pull qwen2.5-coder:7b' on each machine. The built-in demo model still works without it."
  }

  Write-Host "==> Starting MeshPool control plane on port $Port (scale=$Scale)"
  Write-Host "    If Windows asks about network access, choose 'Allow' on Private networks."
  & $Py -m meshpool.controlplane.app
}
finally {
  Pop-Location
}
