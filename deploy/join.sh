#!/usr/bin/env bash
# MeshPool — join this machine to a pool (run on each laptop you want to contribute).
#
# Usage (from the meshpool folder), using the command shown on the control-plane machine:
#   bash deploy/join.sh --server http://192.168.1.20:8080 --token abc123
#   bash deploy/join.sh --server http://192.168.1.20:8080 --token abc123 --cpu 40 --name studio-mac
#
# The agent auto-detects THIS machine's LAN IP and self-registers. Leave it running.
set -euo pipefail

SERVER=""
TOKEN=""
NAME="$(hostname)"
CPU=50
PORT=9001
while [ $# -gt 0 ]; do
  case "$1" in
    --server) SERVER="$2"; shift 2;;
    --token)  TOKEN="$2"; shift 2;;
    --name)   NAME="$2"; shift 2;;
    --cpu)    CPU="$2"; shift 2;;
    --port)   PORT="$2"; shift 2;;
    -h|--help) echo "usage: join.sh --server http://IP:8080 --token XXX [--cpu 50] [--name host] [--port 9001]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done
[ -n "$SERVER" ] || { echo "usage: join.sh --server http://IP:8080 --token XXX" >&2; exit 1; }

cd "$(dirname "$0")/.."
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "Python 3.11+ not found. Install it from https://www.python.org/downloads/ and re-run." >&2; exit 1; }

echo "==> Installing dependencies (first run only)..."
"$PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt

# Preflight: is the control plane reachable? Fail fast with a clear message.
if command -v curl >/dev/null 2>&1; then
  curl -fsS --max-time 5 "$SERVER/api/v1/health" >/dev/null \
    || { echo "Cannot reach control plane at $SERVER. Check the address, that it is running, and its firewall." >&2; exit 1; }
  echo "==> Control plane reachable at $SERVER"
fi

export MESHPOOL_CONTROL_PLANE_URL="$SERVER"
export MESHPOOL_ENROLL_TOKEN="$TOKEN"
export MESHPOOL_NAME="$NAME"
export MESHPOOL_DONATE_CPU_PCT="$CPU"
export MESHPOOL_AGENT_PORT="$PORT"
unset MESHPOOL_AGENT_HOST 2>/dev/null || true   # advertise host auto-detected (LAN IP)

# Friendly Ollama preflight (non-fatal): this machine can serve any Ollama model it holds
# to the whole pool. Show what it will contribute.
if command -v curl >/dev/null 2>&1; then
  OLL="$(curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags 2>/dev/null || true)"
  if [ -n "$OLL" ]; then
    MODELS="$(printf '%s' "$OLL" | grep -o '"name":"[^"]*"' | sed 's/"name":"//;s/"//' | paste -sd, - 2>/dev/null || true)"
    if [ -n "$MODELS" ]; then
      echo "==> Ollama detected here: $MODELS  (these become available to the pool)"
    else
      echo "==> Ollama is running but has no models. Pull one:  ollama pull qwen2.5-coder:7b"
    fi
  else
    echo "==> No local Ollama (optional). To serve a real LLM, install Ollama"
    echo "    (https://ollama.com/download) and 'ollama pull qwen2.5-coder:7b', then re-run this join."
  fi
fi

echo "==> Joining $SERVER as '$NAME' (donating ${CPU}% CPU)"
exec "$PY" -m meshpool.agent.daemon
