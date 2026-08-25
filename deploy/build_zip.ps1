<#
  Build a clean, shareable MeshPool zip - the artifact you hand to anyone so they can
  "unzip and run" (start.cmd / start.sh) with no toolchain.

    powershell -ExecutionPolicy Bypass -File deploy\build_zip.ps1
    powershell -ExecutionPolicy Bypass -File deploy\build_zip.ps1 -Output C:\tmp

  Uses an ALLOWLIST (never a denylist), so runtime state and device keys
  (~/.meshpool, *.db, identity.json, _data_*) can never leak into a shared build.
#>
[CmdletBinding()]
param(
  [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Version from pyproject.toml (single source of truth).
$verLine = Select-String -Path (Join-Path $RepoRoot "pyproject.toml") -Pattern '^\s*version\s*=\s*"([^"]+)"' | Select-Object -First 1
$Version = if ($verLine) { $verLine.Matches[0].Groups[1].Value } else { "0.0.0" }
$Name = "meshpool-$Version"

$OutDir = if ($Output) { $Output } else { Join-Path $RepoRoot "dist" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Stage = Join-Path ([System.IO.Path]::GetTempPath()) ("meshpool-build-" + [guid]::NewGuid().ToString("N"))
$Dest = Join-Path $Stage $Name
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# --- ALLOWLIST: only these ship ------------------------------------------- #
$dirs = @("meshpool", "deploy", "docs")
$files = @("requirements.txt", "pyproject.toml", "README.md", "LICENSE",
           "start.cmd", "start.ps1", "start.sh", "contribute.ps1", "contribute.sh")

foreach ($d in $dirs) {
  $src = Join-Path $RepoRoot $d
  if (Test-Path $src) { Copy-Item -Recurse -Force $src (Join-Path $Dest $d) }
}
foreach ($f in $files) {
  $src = Join-Path $RepoRoot $f
  if (Test-Path $src) { Copy-Item -Force $src (Join-Path $Dest $f) }
}

# --- scrub anything that must never ship (defence in depth) --------------- #
Get-ChildItem -Path $Dest -Recurse -Force -Include "__pycache__" -Directory -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Dest -Recurse -Force -Include "*.pyc", "*.pyo", "*.db", "*.db-wal", "*.db-shm",
  "identity.json", ".meshpool-deps-ok" -File -ErrorAction SilentlyContinue |
  Remove-Item -Force -ErrorAction SilentlyContinue
# never ship the zip-builder itself or any scenario data snapshots
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Dest "deploy\build_zip.ps1")
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Dest "deploy\build_zip.sh")
Get-ChildItem -Path $Dest -Recurse -Force -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like "_data_*" } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$Zip = Join-Path $OutDir "$Name.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $Dest -DestinationPath $Zip -CompressionLevel Optimal
Remove-Item -Recurse -Force $Stage

$sizeMb = [math]::Round((Get-Item $Zip).Length / 1MB, 2)
Write-Host "==> Built $Zip ($sizeMb MB)"
Write-Host "    Share it. Recipients: unzip, then run start.cmd (Windows) or ./start.sh (macOS/Linux)."
