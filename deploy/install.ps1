<#
  MeshPool installer (Windows). Installs Python deps and registers a background service
  (Windows Service via the SCM) for either the control plane or a node agent.

    ./install.ps1 -Role controlplane -Port 8080
    ./install.ps1 -Role agent -Join http://<cp-host>:8080 -Cpu 50

  Run in an elevated (Administrator) PowerShell.
#>
[CmdletBinding()]
param(
  [ValidateSet("controlplane", "agent")] [string]$Role = "agent",
  [int]$Port = 8080,
  [string]$Join = "",
  [int]$Cpu = 50,
  [string]$Name = $env:COMPUTERNAME,
  [int]$AgentPort = 9001,
  [string]$AdvertiseHost = ""
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Py) { $Py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $Py) { throw "Python not found on PATH." }

# The control plane reaches an agent at the address the agent advertises, so on a
# real LAN the agent must advertise a routable IP (not 127.0.0.1). Auto-detect the
# primary IPv4 if the caller didn't pass -AdvertiseHost.
if ($Role -eq "agent" -and -not $AdvertiseHost) {
  try {
    $AdvertiseHost = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
      Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.PrefixOrigin -ne "WellKnown" } |
      Select-Object -First 1).IPAddress
  } catch { }
  if (-not $AdvertiseHost) { $AdvertiseHost = $env:COMPUTERNAME }
}

Write-Host "==> Installing Python dependencies"
& $Py -m pip install --quiet numpy fastapi uvicorn pydantic httpx typer rich pyyaml jinja2

# Bake the role's environment into a wrapper script. `$env:` set in this process
# would NOT survive into a service/task, so the env must live in the wrapper.
if ($Role -eq "controlplane") {
  $svcName = "MeshPoolControlPlane"
  $module  = "meshpool.controlplane.app"
  $envLines = @("set MESHPOOL_PORT=$Port")
} else {
  $svcName = "MeshPoolAgent"
  $module  = "meshpool.agent.daemon"
  if (-not $Join) { throw "Agent role requires -Join http://<control-plane-host>:port" }
  $envLines = @(
    "set MESHPOOL_CONTROL_PLANE_URL=$Join",
    "set MESHPOOL_DONATE_CPU_PCT=$Cpu",
    "set MESHPOOL_NAME=$Name",
    "set MESHPOOL_AGENT_HOST=$AdvertiseHost",
    "set MESHPOOL_AGENT_PORT=$AgentPort"
  )
}

$wrapper = Join-Path $PSScriptRoot "run-$Role.cmd"
$wrapperLines = @("@echo off", "cd /d `"$RepoDir`"") + $envLines + @("`"$Py`" -m $module")
Set-Content -Path $wrapper -Value $wrapperLines -Encoding Ascii

# Register an auto-start background service. `sc.exe create` cannot host `python -m`
# directly — Python is not a service-control binary — so we schedule the wrapper to
# run at startup as SYSTEM (dependency-free; production ships the native Go service).
Write-Host "==> Registering auto-start task $svcName"
schtasks /Create /TN $svcName /TR "`"$wrapper`"" /SC ONSTART /RU SYSTEM /RL HIGHEST /F | Out-Null
schtasks /Run /TN $svcName | Out-Null

Write-Host "==> Done. Dashboard: http://localhost:$Port (control plane)."
if ($Role -eq "agent") { Write-Host "    Advertising this node to the control plane as ${AdvertiseHost}:$AgentPort" }
Write-Host "    Manage with: schtasks /End /TN $svcName  /  schtasks /Delete /TN $svcName /F"
