#!/bin/bash
# ==============================================================================
# Train Y2R Teacher Policy
# ==============================================================================
# Usage:
#   ./scripts/train.sh                              # Fresh training
#   ./scripts/train.sh --continue                   # Resume from latest
#   ./scripts/train.sh --checkpoint path/to/model.pth
# ==============================================================================

source "$(dirname "$0")/common.sh"

parse_checkpoint_args "trajectory" "$@"
shift $PARSED_ARGS

cd "$ISAACLAB_DIR"

Y2R_MODE=train Y2R_TASK=base ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task "$TASK" \
    --headless \
    --track \
    --wandb-project-name trajectory \
    --wandb-entity hgupt3 \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    "$@"

