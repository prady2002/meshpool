#!/usr/bin/env bash
# MeshPool - contribute this machine's spare compute to a hosted MeshPool Cloud.
# The agent dials OUT to the coordinator over a secure tunnel, so it works behind
# home routers / firewalls with nothing to port-forward.
#
#   ./contribute.sh --cloud https://cloud.meshpool.app
#   ./contribute.sh --cloud https://cloud.meshpool.app --cpu 30
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

exec "$PY" -m meshpool.launcher contribute "$@"
