<#
  MeshPool - contribute this machine's spare compute to a hosted MeshPool Cloud.

  The agent dials OUT to the coordinator over a secure tunnel, so it works behind
  home routers / firewalls with nothing to port-forward.

    powershell -ExecutionPolicy Bypass -File contribute.ps1 -Cloud https://cloud.meshpool.app
    powershell -ExecutionPolicy Bypass -File contribute.ps1 -Cloud https://cloud.meshpool.app -Cpu 30
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)] [string]$Cloud,
  [ValidateRange(0, 100)] [int]$Cpu = 50,
  [string]$Name = ""
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
  $a = @("-m", "meshpool.launcher", "contribute", "--cloud", $Cloud, "--cpu", $Cpu)
  if ($Name) { $a += @("--name", $Name) }
  & $Py @a
}
finally { Pop-Location }
