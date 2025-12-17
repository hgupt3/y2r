#!/bin/bash
# Distill teacher policy into student with wrist depth camera
# Usage: ./distill_trajectory.sh TEACHER_CHECKPOINT [--continue] [extra_args...]
# Examples:
#   ./distill_trajectory.sh logs/rl_games/trajectory/.../model.pth
#   ./distill_trajectory.sh path/to/teacher.pth --continue
#   ./distill_trajectory.sh path/to/teacher.pth --beta 0.3
#   ./distill_trajectory.sh path/to/teacher.pth --teacher-agent rl_games_cfg_entry_point

cd /home/harsh/sam/IsaacLab

CHECKPOINT_ARG=""
TEACHER_CKPT=""

# First positional argument is teacher checkpoint
if [ -n "$1" ] && [[ "$1" != --* ]]; then
  TEACHER_CKPT="$1"
  shift
fi

# Check for --continue flag
if [ "$1" == "--continue" ]; then
  LATEST=$(find logs/rl_games/student_depth_distillation -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  
  if [ -z "$LATEST" ]; then
    echo "No checkpoint found in logs/rl_games/student_depth_distillation/"
    echo "Starting fresh distillation."
  else
    echo "========================================"
    echo "Resuming student from: $LATEST"
    echo "========================================"
    CHECKPOINT_ARG="--checkpoint $LATEST"
  fi
  shift
fi

# Validate teacher checkpoint
if [ -z "$TEACHER_CKPT" ]; then
  echo "Error: Teacher checkpoint required"
  echo "Usage: ./distill_trajectory.sh TEACHER_CHECKPOINT [--continue] [extra_args...]"
  echo ""
  echo "Example:"
  echo "  ./distill_trajectory.sh logs/rl_games/trajectory/2024-12-15_12-00-00/nn/trajectory.pth"
  exit 1
fi

if [ ! -f "$TEACHER_CKPT" ]; then
  echo "Error: Teacher checkpoint not found: $TEACHER_CKPT"
  exit 1
fi

echo "========================================"
echo "Teacher checkpoint: $TEACHER_CKPT"
echo "========================================"

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/distill.py \
  --task Isaac-Trajectory-Kuka-Allegro-Student-v0 \
  --teacher-checkpoint "$TEACHER_CKPT" \
  --headless \
  --track \
  --enable_cameras \
  --wandb-project-name distillation \
  --wandb-entity hgupt3 \
  --wandb-name distillation \
  $CHECKPOINT_ARG \
  "$@"
