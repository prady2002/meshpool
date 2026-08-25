#!/usr/bin/env bash
# Build a clean, shareable MeshPool zip (macOS / Linux) - the "unzip and run" artifact.
# Uses an ALLOWLIST, so runtime state and device keys can never leak into a build.
#
#   ./deploy/build_zip.sh
#   ./deploy/build_zip.sh /tmp/out
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/dist}"

VERSION="$(grep -E '^[[:space:]]*version[[:space:]]*=' "$REPO_ROOT/pyproject.toml" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
VERSION="${VERSION:-0.0.0}"
NAME="meshpool-$VERSION"

mkdir -p "$OUT_DIR"
STAGE="$(mktemp -d)"
DEST="$STAGE/$NAME"
mkdir -p "$DEST"

# --- ALLOWLIST: only these ship ------------------------------------------- #
for d in meshpool deploy docs; do
  [ -d "$REPO_ROOT/$d" ] && cp -R "$REPO_ROOT/$d" "$DEST/$d"
done
for f in requirements.txt pyproject.toml README.md LICENSE \
         start.cmd start.ps1 start.sh contribute.ps1 contribute.sh; do
  [ -f "$REPO_ROOT/$f" ] && cp "$REPO_ROOT/$f" "$DEST/$f"
done

# --- scrub anything that must never ship (defence in depth) --------------- #
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DEST" \( -name '*.pyc' -o -name '*.pyo' -o -name '*.db' -o -name '*.db-wal' \
  -o -name '*.db-shm' -o -name 'identity.json' -o -name '.meshpool-deps-ok' \) \
  -type f -delete 2>/dev/null || true
find "$DEST" -type d -name '_data_*' -prune -exec rm -rf {} + 2>/dev/null || true
rm -f "$DEST/deploy/build_zip.ps1" "$DEST/deploy/build_zip.sh" 2>/dev/null || true
chmod +x "$DEST/start.sh" "$DEST/contribute.sh" 2>/dev/null || true

ZIP="$OUT_DIR/$NAME.zip"
rm -f "$ZIP"
( cd "$STAGE" && zip -q -r "$ZIP" "$NAME" )
rm -rf "$STAGE"

echo "==> Built $ZIP"
echo "    Share it. Recipients: unzip, then run start.cmd (Windows) or ./start.sh (macOS/Linux)."
