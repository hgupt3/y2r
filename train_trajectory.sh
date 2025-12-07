#!/bin/bash
# Train trajectory task
# Usage: ./train_trajectory.sh [num_envs] [extra_args...]
# Examples:
#   ./train_trajectory.sh              # Default 16384 envs
#   ./train_trajectory.sh 8192         # Custom envs
#   ./train_trajectory.sh 16384 --track --wandb-project-name traj

cd /home/harsh/sam/IsaacLab

NUM_ENVS=${1:-20480}
shift 2>/dev/null  # Remove first arg if exists, rest goes to extra_args

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Trajectory-Kuka-Allegro-v0 \
  --num_envs $NUM_ENVS \
  --headless \
  --track \
  --wandb-project-name trajectory \
  --wandb-entity hgupt3 \
  --wandb-name trajectory_${NUM_ENVS} \
  "$@"

