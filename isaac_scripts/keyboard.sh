#!/bin/bash
# ==============================================================================
# Keyboard Debug - Palm Orientation Exploration
# ==============================================================================
# Usage:
#   ./isaac_scripts/keyboard.sh                           # ur5e_leap (default)
#   ./isaac_scripts/keyboard.sh --robot kuka_allegro      # kuka_allegro
#   ./isaac_scripts/keyboard.sh --task <name>             # custom task layer
#
# Controls: WASDQE=move, ZXTGCV=rotate, L=reset, ESC=quit
# Palm orientation is printed every ~1 second.
# ==============================================================================

source "$(dirname "$0")/common.sh"

TASK_LAYER="base"

while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
            ROBOT="$2"
            TASK=$(resolve_robot_task "$ROBOT")
            shift 2
            ;;
        --task)
            TASK_LAYER="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

cd "$ISAACLAB_DIR"

echo "========================================"
echo "Keyboard Debug | Robot: $ROBOT | Task: $TASK_LAYER"
echo "========================================"
echo "Controls:"
echo "  W/S     - Move forward/backward (X)"
echo "  A/D     - Move left/right (Y)"
echo "  Q/E     - Move up/down (Z)"
echo "  Z/X     - Roll"
echo "  T/G     - Pitch"
echo "  C/V     - Yaw"
echo "  K       - Toggle gripper open/close"
echo "  L       - Reset keyboard deltas"
echo "  ESC     - Quit"
echo "========================================"

Y2R_MODE=keyboard Y2R_TASK=$TASK_LAYER ./isaaclab.sh -p \
    source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/keyboard_debug.py \
    --task "$TASK" \
    --livestream 2 \
    "$@"

