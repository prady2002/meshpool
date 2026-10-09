#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SERVER=""
TOKEN=""
CPU=50
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server|-s) SERVER="$2"; shift 2 ;;
    --token|-t) TOKEN="$2"; shift 2 ;;
    --cpu|-c) CPU="$2"; shift 2 ;;
    --name|-n) NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$SERVER" ]; then
  echo "Usage: ./join.sh --server http://<coordinator-ip>:8080 --token <TOKEN> --cpu 50"
  exit 1
fi

PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then PY="python"; fi

MARKER="$ROOT/.installed-deps"
if [ ! -f "$MARKER" ]; then
  echo "==> First run: Installing MeshPool..."
  WHL=$(ls "$ROOT"/meshpool-*.whl 2>/dev/null | head -n 1)
  if [ -n "$WHL" ]; then
    "$PY" -m pip install --quiet --disable-pip-version-check "$WHL"
  fi
  touch "$MARKER"
fi

export MESHPOOL_CONTROL_PLANE_URL="$SERVER"
if [ -n "$TOKEN" ]; then export MESHPOOL_INVITE_TOKEN="$TOKEN"; fi
export MESHPOOL_DONATE_CPU_PCT="$CPU"
if [ -n "$NAME" ]; then export MESHPOOL_NAME="$NAME"; fi

exec "$PY" -m meshpool.agent.daemon
