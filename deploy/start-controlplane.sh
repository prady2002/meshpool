#!/usr/bin/env bash
# MeshPool — start the control plane (run this on ONE machine on your LAN).
#
# Usage (from the meshpool folder):
#   bash deploy/start-controlplane.sh
#   bash deploy/start-controlplane.sh --port 8080 --scale home
#
# This machine hosts the dashboard and coordinates the pool. Leave it running.
set -euo pipefail

PORT=8080
SCALE=home
while [ $# -gt 0 ]; do
  case "$1" in
    --port)  PORT="$2"; shift 2;;
    --scale) SCALE="$2"; shift 2;;
    -h|--help) echo "usage: start-controlplane.sh [--port 8080] [--scale home|office|enterprise]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done

cd "$(dirname "$0")/.."
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "Python 3.11+ not found. Install it from https://www.python.org/downloads/ and re-run." >&2; exit 1; }

echo "==> Installing dependencies (first run only)..."
"$PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt

export MESHPOOL_PORT="$PORT"
export MESHPOOL_SCALE="$SCALE"

# Friendly Ollama preflight (non-fatal): MeshPool can route real LLMs to any node that
# runs them under Ollama. Show what THIS machine will contribute.
if command -v curl >/dev/null 2>&1; then
  OLL="$(curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags 2>/dev/null || true)"
  if [ -n "$OLL" ]; then
    MODELS="$(printf '%s' "$OLL" | grep -o '"name":"[^"]*"' | sed 's/"name":"//;s/"//' | paste -sd, - 2>/dev/null || true)"
    if [ -n "$MODELS" ]; then
      echo "==> Ollama detected here: $MODELS"
    else
      echo "==> Ollama is running but has no models. Pull one:  ollama pull qwen2.5-coder:7b"
    fi
  else
    echo "==> No local Ollama (optional). For real LLMs across the pool, install Ollama"
    echo "    (https://ollama.com/download) and 'ollama pull qwen2.5-coder:7b' on each machine."
  fi
fi

echo "==> Starting MeshPool control plane on port $PORT (scale=$SCALE)"
exec "$PY" -m meshpool.controlplane.app
