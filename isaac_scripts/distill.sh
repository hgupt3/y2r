#!/bin/bash
# ==============================================================================
# Distill Y2R Teacher to Student
# ==============================================================================
# Usage:
#   ./scripts/distill.sh --t_continue                           # Latest teacher, fresh student
#   ./scripts/distill.sh --t_checkpoint path/to/teacher.pth     # Specific teacher, fresh student
#   ./scripts/distill.sh --t_continue --continue                # Latest teacher, resume student
#   ./scripts/distill.sh --t_checkpoint t.pth --checkpoint s.pth # Both specific
# ==============================================================================

source "$(dirname "$0")/common.sh"

TEACHER_CHECKPOINT=""
STUDENT_CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
            ROBOT="$2"
            TASK=$(resolve_robot_task "$ROBOT")
            shift 2
            ;;
        --t_continue)
            TEACHER_CHECKPOINT=$(find_latest "trajectory")
            if [ -z "$TEACHER_CHECKPOINT" ]; then
                echo "Error: No teacher checkpoint found in logs/rl_games/trajectory/"
                exit 1
            fi
            echo "========================================"
            echo "Teacher: $TEACHER_CHECKPOINT"
            echo "========================================"
            shift
            ;;
        --t_checkpoint)
            TEACHER_CHECKPOINT="$2"
            require_file "$TEACHER_CHECKPOINT"
            echo "========================================"
            echo "Teacher: $TEACHER_CHECKPOINT"
            echo "========================================"
            shift 2
            ;;
        --continue)
            STUDENT_CHECKPOINT=$(find_latest "student_depth_distillation")
            if [ -z "$STUDENT_CHECKPOINT" ]; then
                echo "No student checkpoint found, starting fresh."
            else
                echo "========================================"
                echo "Resuming student from: $STUDENT_CHECKPOINT"
                echo "========================================"
            fi
            shift
            ;;
        --checkpoint)
            STUDENT_CHECKPOINT="$2"
            require_file "$STUDENT_CHECKPOINT"
            echo "========================================"
            echo "Student: $STUDENT_CHECKPOINT"
            echo "========================================"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Validate teacher checkpoint
if [ -z "$TEACHER_CHECKPOINT" ]; then
    echo "Error: Teacher checkpoint required"
    echo "Usage: ./scripts/distill.sh --t_continue"
    echo "       ./scripts/distill.sh --t_checkpoint path/to/teacher.pth"
    exit 1
fi

cd "$ISAACLAB_DIR"

Y2R_MODE=distill Y2R_TASK=base ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/distill.py \
    --task "$TASK" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --headless \
    --enable_cameras \
    --track \
    --wandb-project-name distillation \
    --wandb-entity hgupt3 \
    ${STUDENT_CHECKPOINT:+--checkpoint "$STUDENT_CHECKPOINT"} \
    "$@"

