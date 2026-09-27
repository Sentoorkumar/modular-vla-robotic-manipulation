#!/bin/bash

# Isaac Sim Launch Script (Clean ROS 2 Environment)
# For: Master Thesis - Pfand Sorting System

# This script launches Isaac Sim with a clean environment
# so that Isaac Sim uses its INTERNAL ROS 2 Humble libraries
# (Python 3.11) instead of the system ROS 2 (Python 3.10).
#
# Usage: ./start_isaacsim.sh


ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$HOME/isaac-sim-5.1}"

# Check if Isaac Sim exists
if [ ! -f "$ISAAC_SIM_PATH/isaac-sim.sh" ]; then
    echo "ERROR: Isaac Sim not found at $ISAAC_SIM_PATH"
    echo "Please update ISAAC_SIM_PATH in this script."
    exit 1
fi

echo "============================================"
echo " Launching Isaac Sim 5.1 with ROS 2 Bridge"
echo " Using: Internal ROS 2 Humble (Python 3.11)"
echo " DDS:   FastDDS (rmw_fastrtps_cpp)"
echo " Domain ID: 0 (default)"
echo "============================================"
echo ""
echo "   source /opt/ros/humble/setup.bash"
echo "   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
echo "   export ROS_DOMAIN_ID=0"
echo ""
echo "============================================"
echo ""

# Launch Isaac Sim in a clean environment
# env -i strips ALL environment variables
# We only pass through what Isaac Sim needs
env -i \
    HOME="$HOME" \
    USER="$USER" \
    DISPLAY="${DISPLAY:-:0}" \
    LANG="${LANG:-en_US.UTF-8}" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
    ROS_DISTRO="humble" \
    RMW_IMPLEMENTATION="rmw_fastrtps_cpp" \
    LD_LIBRARY_PATH="$ISAAC_SIM_PATH/exts/isaacsim.ros2.bridge/humble/lib" \
    bash -c "$ISAAC_SIM_PATH/isaac-sim.sh"
