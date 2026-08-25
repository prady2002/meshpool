<#
  MeshPool - run a complete MeshPool on THIS machine (Windows / PowerShell).

  Just run it (or double-click start.cmd). It installs dependencies on first run,
  starts the control plane + a local agent contributing this machine, and opens the
  dashboard. No IP, port, or token to type.

    powershell -ExecutionPolicy Bypass -File start.ps1
    powershell -ExecutionPolicy Bypass -File start.ps1 -Port 8090 -Cpu 30
#>
[CmdletBinding()]
param(
  [int]$Port = 8080,
  [ValidateRange(0, 100)] [int]$Cpu = 50,
  [string]$Name = "",
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

$Py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $Py) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Py) {
  Write-Host "Python 3.11+ is required. Install it from https://www.python.org/downloads/"
  Write-Host "(tick 'Add python.exe to PATH'), then run this again."
  exit 1
}

Push-Location $Root
try {
  $marker = Join-Path $Root ".meshpool-deps-ok"
  if (-not (Test-Path $marker)) {
    Write-Host "==> First run: installing dependencies (a minute or two) ..."
    & $Py -m pip install --quiet --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Path $marker -Force | Out-Null }
  }
  $a = @("-m", "meshpool.launcher", "solo", "--port", $Port, "--cpu", $Cpu)
  if ($Name) { $a += @("--name", $Name) }
  if ($NoBrowser) { $a += "--no-browser" }
  & $Py @a
}
finally { Pop-Location }
