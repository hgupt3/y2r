#!/bin/bash
# ==============================================================================
# Play/Evaluate Y2R Policy
# ==============================================================================
# Usage:
#   ./scripts/play.sh --continue                        # teacher, base task
#   ./scripts/play.sh --task <name> --continue          # teacher, custom task
#   ./scripts/play.sh --student --continue              # student, base task
#   ./scripts/play.sh --student --task <name> --continue  # student, custom task
#
# Flags can be in any order.
# Tasks are loaded from configs/layers/tasks/<name>.yaml
# To add a new task, create the YAML file and use --task <name>
# ==============================================================================

source "$(dirname "$0")/common.sh"

# Defaults
TASK_LAYER="base"
STUDENT=0
REMAINING_ARGS=()

# Parse all arguments, extract our flags, collect the rest
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_LAYER="$2"
            shift 2
            ;;
        --student)
            STUDENT=1
            shift
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore remaining args for checkpoint parsing
set -- "${REMAINING_ARGS[@]}"

# Parse checkpoint args (--continue or --checkpoint)
if [ "$STUDENT" = "1" ]; then
    parse_checkpoint_args "student_depth_distillation" "$@"
else
    parse_checkpoint_args "trajectory" "$@"
fi
shift $PARSED_ARGS

if [ -z "$CHECKPOINT" ]; then
    echo "Error: Must specify --continue or --checkpoint"
    echo ""
    echo "Usage:"
    echo "  ./scripts/play.sh --continue                          # teacher, base"
    echo "  ./scripts/play.sh --task <name> --continue            # teacher, custom task"
    echo "  ./scripts/play.sh --student --task <name> --continue  # student, custom task"
    echo ""
    echo "Tasks are loaded from configs/layers/tasks/<name>.yaml"
    exit 1
fi

cd "$ISAACLAB_DIR"

if [ "$STUDENT" = "1" ]; then
    echo "========================================"
    echo "Mode: Student | Task: $TASK_LAYER"
    echo "========================================"
    Y2R_MODE=play_student Y2R_TASK=$TASK_LAYER ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
        --task "$TASK" \
        --livestream 2 \
        --enable_cameras \
        --agent rl_games_student_cfg_entry_point \
        --checkpoint "$CHECKPOINT" \
        "$@"
else
    echo "========================================"
    echo "Mode: Teacher | Task: $TASK_LAYER"
    echo "========================================"
    Y2R_MODE=play Y2R_TASK=$TASK_LAYER ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
        --task "$TASK" \
        --livestream 2 \
        --checkpoint "$CHECKPOINT" \
        "$@"
fi
