#!/bin/bash
# ==============================================================================
# Train Y2R Teacher Policy
# ==============================================================================
# Usage:
#   ./isaac_scripts/train.sh                           # Fresh training (MLP)
#   ./isaac_scripts/train.sh --continue                # Resume MLP from latest
#   ./isaac_scripts/train.sh --agent tnet              # Train PointNet+TNet
#   ./isaac_scripts/train.sh --agent tnet --continue   # Resume TNet from latest
#   ./isaac_scripts/train.sh --agent pt                # Train Point Transformer
#   ./isaac_scripts/train.sh --checkpoint path/to/model.pth
#
# Available agents: mlp (default), tnet, pt
# ==============================================================================

source "$(dirname "$0")/common.sh"

parse_agent_args "$@"
shift $PARSED_ARGS

cd "$ISAACLAB_DIR"

Y2R_MODE=train Y2R_TASK=base Y2R_ROBOT=$ROBOT ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task "$TASK" \
    --headless \
    --track \
    --wandb-project-name trajectory \
    --wandb-entity hgupt3 \
    ${AGENT_ARGS} \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    "$@"

