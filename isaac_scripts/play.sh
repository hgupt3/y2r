#!/bin/bash
# ==============================================================================
# Play/Evaluate Y2R Policy
# ==============================================================================
# Usage:
#   ./isaac_scripts/play.sh --continue                     # teacher MLP, base task
#   ./isaac_scripts/play.sh --agent tnet --continue        # teacher TNet, base task
#   ./isaac_scripts/play.sh --task <name> --continue       # teacher, custom task
#   ./isaac_scripts/play.sh --student --continue           # student, base task
#   ./isaac_scripts/play.sh --student --task <name> --continue
#
# Flags can be in any order.
# Available agents: mlp (default), tnet, pt
# Tasks are loaded from configs/layers/tasks/<name>.yaml
# ==============================================================================

source "$(dirname "$0")/common.sh"

# Defaults
TASK_LAYER="base"
STUDENT=0
AGENT_ALIAS=""
REMAINING_ARGS=()

# Parse all arguments, extract our flags, collect the rest
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
        --student)
            STUDENT=1
            shift
            ;;
        --agent)
            AGENT_ALIAS="$2"
            shift 2
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
    # Resolve agent alias for teacher mode
    if [ -n "$AGENT_ALIAS" ]; then
        AGENT_ENTRY=$(resolve_agent_entry "$AGENT_ALIAS")
        AGENT_LOG=$(resolve_agent_log "$AGENT_ALIAS")
        if [ -z "$AGENT_ENTRY" ]; then
            echo "Error: Unknown agent alias '$AGENT_ALIAS'"
            echo "Available agents: mlp, tnet, pt"
            exit 1
        fi
        AGENT_ARGS="--agent $AGENT_ENTRY"
        echo "========================================"
        echo "Agent: $AGENT_ALIAS"
        echo "  Entry point: $AGENT_ENTRY"
        echo "  Log directory: $AGENT_LOG"
        echo "========================================"
        parse_checkpoint_args "$AGENT_LOG" "$@"
    else
        AGENT_ARGS=""
        parse_checkpoint_args "trajectory" "$@"
    fi
fi
shift $PARSED_ARGS

if [ -z "$CHECKPOINT" ]; then
    echo "Error: Must specify --continue or --checkpoint"
    echo ""
    echo "Usage:"
    echo "  ./isaac_scripts/play.sh --continue                     # teacher MLP"
    echo "  ./isaac_scripts/play.sh --agent tnet --continue        # teacher TNet"
    echo "  ./isaac_scripts/play.sh --task <name> --continue       # teacher, custom task"
    echo "  ./isaac_scripts/play.sh --student --continue           # student"
    echo ""
    echo "Available agents: mlp (default), tnet, pt"
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
    echo "Mode: Teacher${AGENT_ALIAS:+ ($AGENT_ALIAS)} | Task: $TASK_LAYER"
    echo "========================================"
    Y2R_MODE=play Y2R_TASK=$TASK_LAYER ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
        --task "$TASK" \
        --livestream 2 \
        ${AGENT_ARGS} \
        --checkpoint "$CHECKPOINT" \
        "$@"
fi
