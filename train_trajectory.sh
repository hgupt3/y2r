#!/bin/bash
# Train trajectory task
# Usage: ./train_trajectory.sh [--continue] [extra_args...]
# Examples:
#   ./train_trajectory.sh              # Fresh training with 20480 envs
#   ./train_trajectory.sh --continue   # Resume from latest checkpoint
#   ./train_trajectory.sh --track --wandb-project-name traj

cd /home/harsh/sam/IsaacLab

CHECKPOINT_ARG=""

# Check for --continue flag
if [ "$1" == "--continue" ]; then
  LATEST=$(find logs/rl_games/trajectory -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  
  if [ -z "$LATEST" ]; then
    echo "No checkpoint found in logs/rl_games/trajectory/"
    echo "Starting fresh training instead."
  else
    echo "========================================"
    echo "Resuming from: $LATEST"
    echo "========================================"
    CHECKPOINT_ARG="--checkpoint $LATEST"
  fi
  shift  # Remove --continue from args
fi

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Trajectory-Kuka-Allegro-v0 \
  --headless \
  --track \
  --wandb-project-name trajectory \
  --wandb-entity hgupt3 \
  --wandb-name trajectory_${NUM_ENVS} \
  $CHECKPOINT_ARG \
  "$@"
