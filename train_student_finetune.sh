#!/bin/bash
# Fine-tune a distilled student policy (wrist depth camera) with PPO.
# Usage:
#   ./train_student_finetune.sh STUDENT_CHECKPOINT [extra_args...]
#   ./train_student_finetune.sh --continue [extra_args...]
#
# Examples:
#   ./train_student_finetune.sh logs/rl_games/student_depth_distillation/.../nn/last_*.pth
#   ./train_student_finetune.sh --continue --num_envs 4096
#   ./train_student_finetune.sh path/to/student.pth --track --wandb-project-name student_ft

cd /home/harsh/sam/IsaacLab

CHECKPOINT_ARG=""
STUDENT_CKPT=""

# First positional argument is student checkpoint (optional unless --continue finds one)
if [ -n "$1" ] && [[ "$1" != --* ]]; then
  STUDENT_CKPT="$1"
  shift
fi

# Check for --continue flag (resume from latest checkpoint)
if [ "$1" == "--continue" ]; then
  LATEST=$(find logs/rl_games/student_depth_distillation -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

  if [ -z "$LATEST" ]; then
    echo "No checkpoint found in logs/rl_games/student_depth_distillation/"
    echo "Starting fine-tuning from scratch (no --checkpoint)."
  else
    echo "========================================"
    echo "Resuming student from: $LATEST"
    echo "========================================"
    CHECKPOINT_ARG="--checkpoint $LATEST"
  fi
  shift
fi

# If a checkpoint was provided explicitly, prefer it
if [ -n "$STUDENT_CKPT" ]; then
  if [ ! -f "$STUDENT_CKPT" ]; then
    echo "Error: Student checkpoint not found: $STUDENT_CKPT"
    exit 1
  fi
  echo "========================================"
  echo "Fine-tuning student from checkpoint: $STUDENT_CKPT"
  echo "========================================"
  CHECKPOINT_ARG="--checkpoint $STUDENT_CKPT"
fi

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Trajectory-Kuka-Allegro-Student-v0 \
  --headless \
  --track \
  --enable_cameras \
  --wandb-project-name student_finetune \
  --wandb-entity hgupt3 \
  --wandb-name student_finetune \
  $CHECKPOINT_ARG \
  "$@"


