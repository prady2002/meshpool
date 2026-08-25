#!/usr/bin/env bash
# MeshPool installer (Linux/macOS). Installs Python deps and registers a background
# service for either the control plane or a node agent.
#
#   sudo ./install.sh --role controlplane --port 8080
#   sudo ./install.sh --role agent --join http://<cp-host>:8080 --cpu 50
set -euo pipefail

ROLE="agent"; PORT="8080"; JOIN=""; CPU="50"; NAME="$(hostname)"; ADVHOST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --role) ROLE="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --join) JOIN="$2"; shift 2;;
    --cpu)  CPU="$2";  shift 2;;
    --name) NAME="$2"; shift 2;;
    --advertise-host) ADVHOST="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3 || command -v python)"

# The control plane reaches an agent at the address the agent advertises, so on a real
# LAN it must be a routable IP, not 127.0.0.1. Auto-detect the primary IPv4 if unset.
if [ "$ROLE" = agent ] && [ -z "$ADVHOST" ]; then
  if hostname -I >/dev/null 2>&1; then ADVHOST="$(hostname -I | awk '{print $1}')"; fi
  if [ -z "$ADVHOST" ] && command -v ipconfig >/dev/null 2>&1; then ADVHOST="$(ipconfig getifaddr en0 2>/dev/null || true)"; fi
  [ -z "$ADVHOST" ] && ADVHOST="$(hostname)"
fi

echo "==> Installing Python dependencies"
"$PY" -m pip install --quiet numpy fastapi uvicorn pydantic httpx typer rich pyyaml jinja2

install_systemd() {
  local unit="$1" role="$2"
  echo "==> Installing systemd unit meshpool-$role"
  sed -e "s#@REPO_DIR@#$REPO_DIR#g" -e "s#@PY@#$PY#g" \
      -e "s#@PORT@#$PORT#g" -e "s#@JOIN@#$JOIN#g" \
      -e "s#@CPU@#$CPU#g" -e "s#@NAME@#$NAME#g" -e "s#@ADVHOST@#$ADVHOST#g" \
      "$REPO_DIR/deploy/systemd/$unit" > "/etc/systemd/system/$unit"
  systemctl daemon-reload
  systemctl enable --now "$unit"
  systemctl --no-pager status "$unit" || true
}

install_launchd() {
  echo "==> Installing launchd agent"
  local plist="$HOME/Library/LaunchAgents/com.meshpool.agent.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  sed -e "s#@REPO_DIR@#$REPO_DIR#g" -e "s#@PY@#$PY#g" \
      -e "s#@JOIN@#$JOIN#g" -e "s#@CPU@#$CPU#g" -e "s#@NAME@#$NAME#g" -e "s#@ADVHOST@#$ADVHOST#g" \
      "$REPO_DIR/deploy/launchd/com.meshpool.agent.plist" > "$plist"
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
}

case "$(uname -s)" in
  Linux)
    if [ "$ROLE" = controlplane ]; then install_systemd meshpool-controlplane.service controlplane
    else install_systemd meshpool-agent.service agent; fi ;;
  Darwin)
    if [ "$ROLE" = controlplane ]; then
      echo "On macOS, run the control plane on an always-on Linux box; installing agent instead."
    fi
    install_launchd ;;
  *) echo "unsupported OS"; exit 1;;
esac

echo "==> Done. Dashboard: http://localhost:${PORT} (control plane)."
