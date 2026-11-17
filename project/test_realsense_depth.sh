#!/usr/bin/env bash
# Open a real window: RGB | depth-overlay | depth heatmap
# (not the ASCII terminal plot from `rs-depth`).
#
# Usage:
#   ./project/test_realsense_depth.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
SCRIPT="$ROOT/project/realsense_overlay.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv python at $PYTHON"
  exit 1
fi

if ! command -v brew >/dev/null || [[ ! -f /opt/homebrew/lib/librealsense2.dylib ]]; then
  echo "Homebrew librealsense not found. Install with:"
  echo "  brew install librealsense"
  exit 1
fi

echo "Opening color + depth overlay window (sudo for USB on macOS)…"
echo "Keys: q quit | s save | [ ] overlay strength"
exec sudo -E "$PYTHON" "$SCRIPT"
