#!/bin/bash
# ROS 2 Terminal Setup Script
# For: Master Thesis - Sorting System
# Run this in every new terminal where you want
# to use ROS 2 nodes (perception, LLM, STT, etc.)
# Usage: source setup_ros2_terminal.sh
# (Must use 'source' not './' so variables persist)

source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# Source your workspace
if [ -f "$HOME/llm_arm_ws/install/setup.bash" ]; then
    source "$HOME/llm_arm_ws/install/setup.bash"
    echo "Workspace sourced: ~/llm_arm_ws"
fi

echo "ROS 2 Humble ready"
echo "DDS: FastDDS | Domain ID: 0"
echo ""
echo "You can now run your ROS 2 nodes:"
echo "  ros2 topic list"
echo "  etc."
