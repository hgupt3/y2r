#!/bin/bash
# View latest trajectory checkpoint via WebRTC livestream
# Auto-finds most recent checkpoint in logs/rl_games/trajectory/
# Connect via browser at http://localhost:8211/streaming/webrtc-client

cd /home/harsh/sam/IsaacLab

# Find latest checkpoint
LATEST=$(find logs/rl_games/trajectory -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST" ]; then
  echo "No checkpoint found in logs/rl_games/trajectory/"
  echo "Make sure you've run training first."
  exit 1
fi

echo "========================================"
echo "Loading checkpoint: $LATEST"
echo "========================================"

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Trajectory-Kuka-Allegro-PushT-v0 \
  --checkpoint "$LATEST" \
  --headless \
  --livestream 2 \
