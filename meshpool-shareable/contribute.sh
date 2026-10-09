#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then PY="python"; fi

CLOUD=""
CPU=50
NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cloud) CLOUD="$2"; shift 2 ;;
    --cpu) CPU="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ -z "$CLOUD" ]; then
  echo "Usage: ./contribute.sh --cloud https://cloud.meshpool.app --cpu 50"
  exit 1
fi

MARKER="$ROOT/.installed-deps"
if [ ! -f "$MARKER" ]; then
  WHL=$(ls "$ROOT"/meshpool-*.whl 2>/dev/null | head -n 1)
  if [ -n "$WHL" ]; then "$PY" -m pip install --quiet --disable-pip-version-check "$WHL"; fi
  touch "$MARKER"
fi

ARGS=("-m" "meshpool.launcher" "contribute" "--cloud" "$CLOUD" "--cpu" "$CPU")
if [ -n "$NAME" ]; then ARGS+=("--name" "$NAME"); fi
exec "$PY" "${ARGS[@]}"
