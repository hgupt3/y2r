#!/bin/bash
# Activate sam conda environment + ROS2 Humble
# Usage: source activate_ros.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam

# Source ROS2
source /opt/ros/humble/setup.bash

# Source custom message types
source /home/harsh/sam/real_world_execution/install/setup.bash

echo "✓ Environment ready:"
echo "  Python: $(python --version)"
echo "  ROS2: Humble"
echo "  Conda env: sam"
echo "  Custom interfaces: tracker_interfaces"
echo ""
echo "Test with: ros2 topic list"

