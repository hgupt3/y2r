#!/bin/bash
# ==============================================================================
# Test URDF → USD conversion for UR5e + LEAP Hand
# ==============================================================================
# Usage:
#   ./isaac_scripts/generate_usd.sh            # Headless report only
#   ./isaac_scripts/generate_usd.sh --view     # Open browser viewer at localhost:8211
# ==============================================================================

source "$(dirname "$0")/common.sh"

cd "$ISAACLAB_DIR"

./isaaclab.sh -p "$REPO_ROOT/isaac_scripts/generate_usd.py" --headless "$@"
