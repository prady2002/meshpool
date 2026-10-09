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
if (-not $Py) { Write-Host "Python 3.11+ required."; exit 1 }

Push-Location $Root
try {
  $marker = Join-Path $Root ".installed-deps"
  $whlItem = Get-ChildItem -Path $Root -Filter "meshpool-*.whl" | Select-Object -First 1
  if (-not (Test-Path $marker)) {
    if ($whlItem) { & $Py -m pip install --quiet --disable-pip-version-check $whlItem.FullName }
    if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Path $marker -Force | Out-Null }
  }
  $a = @("-m", "meshpool.launcher", "contribute", "--cloud", $Cloud, "--cpu", $Cpu)
  if ($Name) { $a += @("--name", $Name) }
  & $Py @a
}
finally { Pop-Location }
