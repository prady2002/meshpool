#!/usr/bin/env bash
# MeshPool - run a complete MeshPool on THIS machine (macOS / Linux).
# Installs dependencies on first run, starts the control plane + a local agent
# contributing this machine, and opens the dashboard. No IP/port/token to type.
#
#   ./start.sh                 # run on the default port (8080)
#   ./start.sh --port 8090 --cpu 30
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "Python 3.11+ is required. Install it from https://www.python.org/downloads/"
  echo "(or your package manager), then run this again."
  exit 1
fi

if [ ! -f ".meshpool-deps-ok" ]; then
  echo "==> First run: installing dependencies (a minute or two) ..."
  "$PY" -m pip install --quiet --disable-pip-version-check -r requirements.txt && touch ".meshpool-deps-ok"
fi

exec "$PY" -m meshpool.launcher solo "$@"
