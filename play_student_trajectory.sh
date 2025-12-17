#!/bin/bash
# Play/evaluate a trained student policy with wrist depth camera
# Usage: ./play_student_trajectory.sh CHECKPOINT [extra_args...]
# Examples:
#   ./play_student_trajectory.sh logs/rl_games/student_depth_distillation/2024-12-16_12-00-00/nn/student_depth_distillation_distill_5000.pth
#   ./play_student_trajectory.sh path/to/checkpoint.pth --num_envs 16

cd /home/harsh/sam/IsaacLab

CHECKPOINT=""

# First positional argument is checkpoint
if [ -n "$1" ] && [[ "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

# If no checkpoint provided, try to find the latest one
if [ -z "$CHECKPOINT" ]; then
  LATEST=$(find logs/rl_games/student_depth_distillation -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  
  if [ -z "$LATEST" ]; then
    echo "Error: No checkpoint provided and no checkpoint found in logs/rl_games/student_depth_distillation/"
    echo ""
    echo "Usage: ./play_student_trajectory.sh CHECKPOINT [extra_args...]"
    echo ""
    echo "Example:"
    echo "  ./play_student_trajectory.sh logs/rl_games/student_depth_distillation/2024-12-16_12-00-00/nn/checkpoint.pth"
    exit 1
  fi
  
  CHECKPOINT="$LATEST"
  echo "Using latest checkpoint: $CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "Error: Checkpoint not found: $CHECKPOINT"
  exit 1
fi

echo "========================================"
echo "Playing student policy"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Trajectory-Kuka-Allegro-Student-Play-v0 \
  --checkpoint "$CHECKPOINT" \
  --headless \
  --livestream 2 \
  --enable_cameras \
  "$@"

