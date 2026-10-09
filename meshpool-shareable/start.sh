#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then PY="python"; fi
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.11+ is required. Install from https://www.python.org/downloads/"
  exit 1
fi

MARKER="$ROOT/.installed-deps"
if [ ! -f "$MARKER" ]; then
  echo "==> First run: Installing MeshPool build package..."
  WHL=$(ls "$ROOT"/meshpool-*.whl 2>/dev/null | head -n 1)
  if [ -n "$WHL" ]; then
    "$PY" -m pip install --quiet --disable-pip-version-check "$WHL"
  fi
  touch "$MARKER"
fi

exec "$PY" -m meshpool.launcher solo "$@"
