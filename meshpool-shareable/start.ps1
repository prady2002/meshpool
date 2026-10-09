<#
  MeshPool - Instant One-Click Launcher (Windows / PowerShell)
  Installs the pre-built MeshPool build artifact (wheel) and launches the application.
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
  Write-Host "(Make sure to check 'Add python.exe to PATH'), then run this again."
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
    } else {
      & $Py -m pip install --quiet --disable-pip-version-check meshpool
    }
    if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Path $marker -Force | Out-Null }
  }
  $a = @("-m", "meshpool.launcher", "solo", "--port", $Port, "--cpu", $Cpu)
  if ($Name) { $a += @("--name", $Name) }
  if ($NoBrowser) { $a += "--no-browser" }
  & $Py @a
}
finally { Pop-Location }
