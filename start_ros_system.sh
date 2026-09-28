#!/bin/bash

# ============================================================
# Modular VLA Robotic Manipulation System
# ROS 2 Multi-Terminal Launcher
# ============================================================
#
# Opens each ROS 2 component in a separate GNOME Terminal.
# Every terminal sources the project's ROS 2 environment before
# starting its node.
#
# IMPORTANT:
# 1. Start Isaac Sim separately using ./start_isaacsim.sh
# 2. Open the provided USD scene and press Play.
# 3. Run grasp_with_judge.py in the Isaac Sim Script Editor.
# 4. Then run this script.
# ============================================================

WORKSPACE="$HOME/llm_arm_ws"
SETUP_SCRIPT="$WORKSPACE/setup_ros2_terminal.sh"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "ERROR: gnome-terminal is not installed."
    exit 1
fi

if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "ERROR: ROS 2 setup script not found:"
    echo "  $SETUP_SCRIPT"
    exit 1
fi

if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: Workspace not found:"
    echo "  $WORKSPACE"
    exit 1
fi

echo "============================================================"
echo " Starting Modular VLA ROS 2 System"
echo "============================================================"
echo
echo "Workspace: $WORKSPACE"
echo
echo "Make sure:"
echo "  - Isaac Sim is running"
echo "  - The USD scene is open"
echo "  - Simulation is in Play mode"
echo "  - grasp_with_judge.py has been run in the Script Editor"
echo
echo "Opening ROS 2 terminals..."
echo

# ------------------------------------------------------------
# 1. Speech-to-Text
# ------------------------------------------------------------

echo "[1/7] Starting Speech-to-Text..."

gnome-terminal \
    --title="VLA - 1 - Speech to Text" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== Speech-to-Text Node ===';
        ros2 run speech_interface stt_node;
        exec bash
    "

sleep 2

# ------------------------------------------------------------
# 2. Trajectory Bridge
# ------------------------------------------------------------

echo "[2/7] Starting Trajectory Bridge..."

gnome-terminal \
    --title="VLA - 2 - Trajectory Bridge" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== Trajectory Bridge ===';
        ros2 run motion_agent trajectory_bridge --ros-args \
            -p use_sim_time:=true \
            -p auto_execute:=true;
        exec bash
    "

sleep 2

# ------------------------------------------------------------
# 3. MoveIt 2
# ------------------------------------------------------------

echo "[3/7] Starting MoveIt 2..."

gnome-terminal \
    --title="VLA - 3 - MoveIt 2" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== MoveIt 2 ===';
        ros2 launch piper_moveit_config isaac_integrated.launch.py;
        exec bash
    "

# MoveIt requires additional startup time before motion requests.
echo "Waiting for MoveIt to initialize..."
sleep 10

# ------------------------------------------------------------
# 4. Perception Agent
# ------------------------------------------------------------

echo "[4/7] Starting Perception Agent..."

gnome-terminal \
    --title="VLA - 4 - Perception Agent" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== Perception Agent ===';
        ros2 run perception_agent perception_node;
        exec bash
    "

sleep 2

# ------------------------------------------------------------
# 5. Motion Agent
# ------------------------------------------------------------

echo "[5/7] Starting Motion Agent..."

gnome-terminal \
    --title="VLA - 5 - Motion Agent" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== Motion Agent ===';
        ros2 run motion_agent motion_agent_node;
        exec bash
    "

sleep 2

# ------------------------------------------------------------
# 6. LLM Coordinator
# ------------------------------------------------------------

echo "[6/7] Starting LLM Coordinator..."

gnome-terminal \
    --title="VLA - 6 - LLM Coordinator" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== LLM Coordinator ===';
        ros2 run llm_coordinator llm_node;
        exec bash
    "

sleep 2

# ------------------------------------------------------------
# 7. Text-to-Speech
# ------------------------------------------------------------

echo "[7/7] Starting Text-to-Speech..."

gnome-terminal \
    --title="VLA - 7 - Text to Speech" \
    -- bash -c "
        cd '$WORKSPACE';
        source '$SETUP_SCRIPT';
        echo;
        echo '=== Text-to-Speech Node ===';
        ros2 run speech_interface tts_node;
        exec bash
    "

echo
echo "============================================================"
echo " All ROS 2 terminals have been started."
echo "============================================================"
echo
echo "Startup order:"
echo "  1. Speech-to-Text"
echo "  2. Trajectory Bridge"
echo "  3. MoveIt 2"
echo "  4. Perception Agent"
echo "  5. Motion Agent"
echo "  6. LLM Coordinator"
echo "  7. Text-to-Speech"
echo
