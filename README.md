# Modular Vision-Language-Action System for Voice-Controlled Robotic Manipulation

A modular ROS 2 framework for voice-controlled robotic manipulation in NVIDIA Isaac Sim. The system combines speech recognition, language-based task coordination, visual perception, motion planning, and speech feedback to enable a simulated AgileX Piper robot to identify, grasp, and sort German Pfand objects using natural-language commands.

The project is designed as a modular Vision-Language-Action (VLA) pipeline in which perception, reasoning, and robot execution remain separate ROS 2 components.

## System Overview

The processing pipeline is:

```text
Voice Input
    ↓
Whisper STT
    ↓
LLM Coordinator
(Qwen2.5 + Knowledge/Rule Layer)
    ↓
Perception Agent
(Custom YOLOv8 + Pixel-to-World Estimation)
    ↓
Motion Agent
(MoveIt 2 + TRAC-IK)
    ↓
AgileX Piper Robot
(NVIDIA Isaac Sim)
    ↓
Piper TTS Feedback
```

The system supports commands such as selecting a specific object, selecting an object class, resolving ambiguous references, selecting objects using spatial descriptions, and clearing multiple objects from the workspace.

---

## Main Components

### 1. Speech Interface

The speech interface provides both speech-to-text and text-to-speech communication.

**Speech-to-Text**

OpenAI Whisper is used for local speech recognition. The STT node records the user's command and publishes the recognized text to the ROS 2 system.

Executable:

```bash
ros2 run speech_interface stt_node
```

**Text-to-Speech**

Piper TTS provides local spoken feedback from the robot system.

Executable:

```bash
ros2 run speech_interface tts_node
```

The current configuration uses:

```text
en_US-lessac-medium
```

---

### 2. LLM Coordinator

The LLM coordinator interprets natural-language commands and coordinates perception and manipulation.

The coordinator uses a hybrid architecture:

- deterministic rules for common and safety-critical decisions;
- a knowledge base containing sorting, safety, priority, clarification, and feedback rules;
- Qwen2.5 through Ollama for language interpretation when additional reasoning is required;
- structured ROS 2 commands for communication with the motion system.

The coordinator also handles ambiguous commands. For example, if several bottles are visible and the user requests:

```text
pick the bottle
```

the system can request clarification instead of selecting an object arbitrarily.

The currently tested local model is:

```text
qwen2.5:7b-instruct-q4_0
```

Executable:

```bash
ros2 run llm_coordinator llm_node
```

---

### 3. Perception Agent

The perception agent detects Pfand objects and estimates their positions in the simulated workspace.

A custom YOLOv8n model is used to detect two object classes:

```text
bottle
can
```

The detector was trained using synthetic images generated in NVIDIA Isaac Sim.

The runtime model is included in:

```text
src/perception_agent/models/best.pt
```

Object image coordinates are converted to robot workspace coordinates using camera calibration and ray-plane intersection.

The perception system also generates object identifiers such as:

```text
bottle 1
bottle 2
can 1
can 2
```

These identifiers allow the language coordinator to refer to individual detected objects.

Executable:

```bash
ros2 run perception_agent perception_node
```

---

### 4. Motion Agent

The motion agent converts the selected target into robot manipulation actions.

It uses:

- MoveIt 2 for motion planning;
- TRAC-IK for inverse kinematics;
- target positions supplied by the perception system;
- predefined destination poses for object sorting.

Different grasp strategies are used depending on object orientation.

Standing objects use a side-grasp strategy, while lying objects use a top-down grasp strategy.

Executable:

```bash
ros2 run motion_agent motion_agent_node
```

A trajectory bridge transfers planned trajectories to the simulated Piper robot:

```bash
ros2 run motion_agent trajectory_bridge --ros-args -p use_sim_time:=true -p auto_execute:=true
```

---

### 5. Isaac Sim Grasp Handling

The file

```text
grasp_with_judge.py
```

handles simulation-specific grasp attachment and optional evaluation functionality.

Because stable friction-only grasping was not sufficiently reliable with the simulated gripper configuration, the object is temporarily attached to the gripper during a successful grasp and released during placement.

This mechanism is specific to the simulation environment and should not be interpreted as a hardware grasp-control method.

**Important:** `grasp_with_judge.py` is not launched as a ROS 2 node.

It must be opened and executed manually inside the **NVIDIA Isaac Sim Script Editor** after the simulation starts.

If the Isaac Sim timeline is stopped and started again, run the script again.

---

## Repository Structure

```text
modular-vla-robotic-manipulation/
│
├── src/
│   ├── speech_interface/
│   │   ├── stt_node.py
│   │   └── tts_node.py
│   │
│   ├── llm_coordinator/
│   │   ├── llm_node.py
│   │   └── knowledge/
│   │
│   ├── perception_agent/
│   │   ├── perception_node.py
│   │   ├── pixel_to_world.py
│   │   ├── config/
│   │   └── models/
│   │       └── best.pt
│   │
│   ├── motion_agent/
│   │   ├── motion_agent_node.py
│   │   ├── trajectory_bridge.py
│   │   └── metrics_logger.py
│   │
│   ├── robot_interfaces/
│   │
│   └── piper_moveit_config/
│
├── training/
│   └── YOLO training resources
│
├── evaluation/
│   └── Optional evaluation and analysis resources
│
├── grasp_with_judge.py
├── setup_ros2_terminal.sh
├── start_isaacsim.sh
├── .gitignore
└── README.md
```

Some large or third-party resources are intentionally not stored directly in the Git repository. Their installation is described below.

---

# Installation

## 1. Tested Environment

The project was developed and tested with:

```text
Ubuntu 22.04
ROS 2 Humble
Python 3.10
NVIDIA Isaac Sim 5.1
NVIDIA RTX 3500 Ada Generation Laptop GPU
```

The project uses Fast DDS with:

```text
ROS_DOMAIN_ID=0
```

A CUDA-capable NVIDIA GPU is strongly recommended for Isaac Sim and neural-network inference.

---

## 2. Clone the Repository

The current configuration expects the workspace to be located at:

```text
~/llm_arm_ws
```

Clone the repository using:

```bash
cd ~
git clone https://github.com/Sentoorkumar/modular-vla-robotic-manipulation.git llm_arm_ws
cd ~/llm_arm_ws
```

---

## 3. Install ROS 2 Dependencies

Install ROS 2 Humble according to the official ROS 2 documentation before continuing.

Install the required development tools:

```bash
sudo apt update
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    ffmpeg \
    alsa-utils \
    libnlopt-dev \
    libnlopt-cxx-dev
```

If `rosdep` has not previously been initialized:

```bash
sudo rosdep init
rosdep update
```

Then install available ROS package dependencies:

```bash
cd ~/llm_arm_ws
rosdep install --from-paths src --ignore-src -r -y
```

---

## 4. Install Python Dependencies

Install the Python packages required by the perception, speech, and coordinator components:

```bash
python3 -m pip install \
    ultralytics \
    opencv-python \
    numpy \
    scipy \
    requests \
    sounddevice \
    openai-whisper \
    piper-tts
```

PyTorch must also be available for Whisper and YOLO inference. A CUDA-enabled PyTorch installation is recommended when using an NVIDIA GPU.

---

## 5. Install the AgileX Piper Isaac Sim Resources

The Piper robot description is obtained from the AgileX Piper Isaac Sim repository.

```bash
cd ~/llm_arm_ws

git clone https://github.com/agilexrobotics/piper_isaac_sim.git \
    external_resources/piper_isaac_sim

cd external_resources/piper_isaac_sim

git checkout 8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2
```

Create the required symbolic link:

```bash
cd ~/llm_arm_ws

ln -s \
    "$HOME/llm_arm_ws/external_resources/piper_isaac_sim/piper_description" \
    src/piper_description
```

The `external_resources/` directory and `src/piper_description` symbolic link are intentionally excluded from Git.

---

## 6. Install TRAC-IK

TRAC-IK is required by the MoveIt configuration used in this project.

Clone it into the ROS 2 workspace:

```bash
cd ~/llm_arm_ws/src

git clone https://bitbucket.org/traclabs/trac_ik.git

cd trac_ik

git checkout 2933c62ecc641cc8c242cdfd102205da38165359
```

Return to the workspace:

```bash
cd ~/llm_arm_ws
```

The `src/trac_ik/` directory is intentionally excluded from this repository because it is a third-party dependency.

---

## 7. Install Ollama and Qwen2.5

Install Ollama using the installation instructions provided by the official Ollama project.

After Ollama is installed, download the model used by the coordinator:

```bash
ollama pull qwen2.5:7b-instruct-q4_0
```

Verify that the model is available:

```bash
ollama list
```

The coordinator communicates with the local Ollama service.

No external cloud LLM API is required during normal operation.

---

## 8. Configure Piper TTS

The TTS node uses Piper with the following voice:

```text
en_US-lessac-medium
```

The required files are:

```text
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json
```

Place both files in:

```text
~/.local/share/piper-voices/
```

The final paths should therefore be:

```text
~/.local/share/piper-voices/en_US-lessac-medium.onnx
~/.local/share/piper-voices/en_US-lessac-medium.onnx.json
```

Verify them with:

```bash
ls -lh ~/.local/share/piper-voices/
```

---

## 9. Download the Isaac Sim Scene

The complete Isaac Sim scene is distributed separately as a GitHub Release asset because the scene and its collected assets are too large to keep in the main Git repository.

Download:

```text
main_scene.zip
```

from the **Releases** section of this repository:

https://github.com/Sentoorkumar/modular-vla-robotic-manipulation/releases

Create the scene directory:

```bash
cd ~/llm_arm_ws
mkdir -p isaac_scene
```

Extract `main_scene.zip` into:

```text
~/llm_arm_ws/isaac_scene/
```

After extraction, the main scene should be available under:

```text
~/llm_arm_ws/isaac_scene/Collected_main_scene_with_piper/
```

Open the main USD scene from this directory in Isaac Sim.

---

## 10. Configure the Isaac Sim Path

The launch script expects Isaac Sim at:

```text
~/isaac-sim-5.1
```

If Isaac Sim is installed elsewhere, define `ISAAC_SIM_PATH` before launching:

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
```

The script uses the value of `ISAAC_SIM_PATH` when it is defined.

---

## 11. Build the ROS 2 Workspace

Build the complete workspace:

```bash
cd ~/llm_arm_ws

source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

colcon build --symlink-install
```

After a successful build, the required ROS 2 packages should be available.

---

# Running the System

## Recommended: Automatic ROS 2 Startup

For normal operation, the ROS 2 components can be started automatically using the provided launcher script. The script opens each component in a separate GNOME Terminal and sources the ROS 2 workspace environment automatically in every terminal.

Before running the launcher:

1. Start NVIDIA Isaac Sim:

   ```bash
   cd ~/llm_arm_ws
   ./start_isaacsim.sh
   ```

2. Open the supplied main USD scene.
3. Press **PLAY** in Isaac Sim.
4. Open the Isaac Sim **Script Editor** and run `grasp_with_judge.py`.

   **Important:** If the simulation is stopped and PLAY is pressed again, run `grasp_with_judge.py` again.

5. Make sure the Ollama service is available and the required Qwen model has been downloaded.

Then open a normal Ubuntu terminal and run:

```bash
cd ~/llm_arm_ws
./start_ros_system.sh
```

The launcher starts the following ROS 2 components in separate terminal windows:

1. Speech-to-Text
2. Trajectory Bridge
3. MoveIt 2
4. Perception Agent
5. Motion Agent
6. LLM Coordinator
7. Text-to-Speech

A startup delay is included between components, with additional initialization time provided for MoveIt 2.

Once all terminals are running, the system is ready to receive spoken commands through the Speech-to-Text node.

The manual startup procedure below can be used for debugging, development, or starting individual components separately.

---

## Manual Startup


## Important: Initialize Every ROS 2 Terminal

For every new terminal used for a ROS 2 node, first run:

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh
```

This configures ROS 2 Humble, the workspace, Fast DDS, and the ROS domain.

---

## Terminal 1 — Start NVIDIA Isaac Sim

```bash
cd ~/llm_arm_ws
./start_isaacsim.sh
```

After Isaac Sim opens:

1. Open the supplied main scene.
2. Press **PLAY**.
3. Open the Isaac Sim **Script Editor**.
4. Open or paste `grasp_with_judge.py`.
5. Run the script.

The console should indicate that the grasp helper is running.

**Important:** If the simulation is stopped and PLAY is pressed again, run `grasp_with_judge.py` again in the Script Editor.

---

## Terminal 2 — Start the Trajectory Bridge

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 run motion_agent trajectory_bridge \
    --ros-args \
    -p use_sim_time:=true \
    -p auto_execute:=true
```

Keep this terminal running.

---

## Terminal 3 — Start MoveIt 2

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 launch piper_moveit_config isaac_integrated.launch.py
```

Wait until MoveIt reports that planning can begin before continuing.

---

## Terminal 4 — Start the Perception Agent

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 run perception_agent perception_node
```

The node should begin receiving the Isaac Sim camera stream and publishing detected objects.

---

## Terminal 5 — Start the Motion Agent

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 run motion_agent motion_agent_node
```

---

## Terminal 6 — Start the LLM Coordinator

Make sure the Ollama service is available and that the Qwen model has already been downloaded.

Then run:

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 run llm_coordinator llm_node
```

---

# Testing Without Voice

The complete perception-language-motion pipeline can be tested without starting the microphone or TTS nodes.

Open another terminal:

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh
```

For example:

```bash
ros2 topic pub --once /voice_cmd std_msgs/msg/String "{data: 'clear all'}"
```

This is useful for testing the robotic pipeline independently of speech recognition.

Other natural-language commands can also be published through `/voice_cmd`.

---

# Running the Full Voice-Controlled System

After the core system is running, start the speech nodes in separate terminals.

## Terminal 7 — Speech-to-Text

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 run speech_interface stt_node
```

Speak a command when prompted.

---

## Terminal 8 — Text-to-Speech

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh

ros2 run speech_interface tts_node
```

The TTS node converts system feedback into spoken responses using Piper.

---

# Main ROS 2 Interfaces

The main communication flow includes:

```text
Voice / Text Command
        ↓
    /voice_cmd
        ↓
LLM Coordinator
        ↕
/perception/result
        ↓
Structured Robot Command
        ↓
Motion Agent
        ↓
MoveIt / Trajectory Bridge
        ↓
Isaac Sim Piper Robot
```

Important interfaces include:

| Interface | Purpose |
|---|---|
| `/voice_cmd` | Natural-language command input |
| `/perception/result` | Detected objects and estimated workspace positions |
| `/llm/command` | Structured command from the coordinator |
| `/feedback/text` | Text feedback for the user |
| `/joint_command` | Robot joint commands used by the Isaac Sim interface |

The custom ROS 2 messages used by the system are defined in:

```text
src/robot_interfaces/
```

---

# Knowledge-Grounded Coordination

The coordinator combines natural-language interpretation with explicit task knowledge rather than relying entirely on unrestricted LLM output.

The knowledge layer contains rules related to:

- object sorting;
- reachability and safety;
- target prioritization;
- ambiguity handling;
- clarification;
- user feedback.

Common commands can therefore be processed deterministically, while the local language model is used selectively when linguistic interpretation is required.

This architecture keeps robot execution separated from free-form language-model output.

---

# Object Selection and Clarification

The system can distinguish multiple objects of the same class using generated labels such as:

```text
bottle 1
bottle 2
can 1
can 2
can 3
```

When a command does not uniquely identify an object, the coordinator can request additional information.

The perception system can also support spatial references based on the detected scene, such as selecting an object according to its relative position or distance from the robot.

Before execution, the coordinator applies checks including object availability, perception freshness, and reachability.

---

# YOLO Dataset and Training

The object detector was trained on a synthetic dataset generated using NVIDIA Isaac Sim Replicator.

The dataset contains the two classes:

```text
bottle
can
```

The dataset is available separately on Kaggle:

https://kaggle.com/datasets/37d4d45318e4db03f9f9244173164ba06919a4f76a3f6e16e55013e8b969f834

Training-related resources are stored under:

```text
training/
```

The trained model used during runtime is:

```text
src/perception_agent/models/best.pt
```

The base Ultralytics pretrained models are intentionally not stored in the repository.

---

# Optional Evaluation Utilities

The files under:

```text
evaluation/
```

are **not required to run the normal voice-controlled sorting system**.

They are provided for optional evaluation, debugging, and reproducibility.

Similarly, the evaluation functions contained in:

```text
grasp_with_judge.py
```

are optional.

For normal operation, only the grasp helper functionality needs to be running in the Isaac Sim Script Editor.

Optional evaluation can record manipulation outcomes and other measurements for later analysis.

An evaluation session can use functions such as:

```python
snapshot_start("standing")
```

or:

```python
snapshot_start("lying")
```

After the corresponding manipulation command has completed, the trial can be evaluated using:

```python
judge()
```

Evaluation logs can be stored using the default metrics location or a path configured through:

```text
THESIS_METRICS_FILE
```

Existing evaluation data can be analysed with:

```bash
cd ~/llm_arm_ws

python3 evaluation/analyze_metrics.py \
    evaluation/run_log_standing.csv
```

Again, none of these evaluation steps are required for normal system startup or operation.

---

# Important Notes and Limitations

### Simulation-specific grasp attachment

The current grasp attachment mechanism is designed for Isaac Sim. It temporarily constrains the grasped object to the robot during manipulation.

A physical robot implementation would require appropriate gripper control, contact handling, force control, and hardware-specific safety mechanisms.

### Object classes

The supplied detector currently supports:

```text
bottle
can
```

Additional object categories require corresponding training data and detector retraining.

### Lying-object grasping

Lying objects use a top-down grasp strategy with a predefined wrist orientation. This approach has a more constrained reachable workspace than the standing-object grasp strategy.

### Camera visibility

The system can only reason about objects visible to the perception camera. Severe occlusion can prevent an object from being detected or localized correctly.

### Local execution

The major runtime components are designed to operate locally after the required models and dependencies have been installed.

---

# Troubleshooting

## ROS 2 package not found

Make sure the workspace has been built and the terminal initialized:

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh
```

---

## Isaac Sim does not start

Check the configured Isaac Sim path:

```bash
echo $ISAAC_SIM_PATH
```

If necessary:

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim
```

Then:

```bash
cd ~/llm_arm_ws
./start_isaacsim.sh
```

---

## Piper robot description is missing

Check:

```bash
ls -l ~/llm_arm_ws/src/piper_description
```

It should point to:

```text
~/llm_arm_ws/external_resources/piper_isaac_sim/piper_description
```

---

## TRAC-IK is missing

Check:

```bash
ls ~/llm_arm_ws/src/trac_ik
```

If it does not exist, repeat the TRAC-IK installation instructions above and rebuild the workspace.

---

## Qwen model is missing

Check:

```bash
ollama list
```

If necessary:

```bash
ollama pull qwen2.5:7b-instruct-q4_0
```

---

## Piper voice is missing

Check:

```bash
ls ~/.local/share/piper-voices/
```

The directory should contain:

```text
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json
```

---

## No robot motion

Verify that:

1. Isaac Sim is running and the timeline is playing.
2. `grasp_with_judge.py` has been executed after the latest PLAY.
3. `trajectory_bridge` is running with `auto_execute:=true`.
4. MoveIt 2 has completed initialization.
5. The perception node is publishing detections.
6. The motion agent is running.
7. The requested object is visible and reachable.

---

# Normal Startup Summary

Once the project has been installed and built, the normal startup sequence is:

```text
1. Start Isaac Sim
2. Open the supplied scene and press PLAY
3. Run grasp_with_judge.py in the Isaac Sim Script Editor
4. Start trajectory_bridge
5. Start MoveIt 2
6. Start perception_agent
7. Start motion_agent
8. Start llm_coordinator
9. Optionally start stt_node and tts_node
10. Send or speak a command
```

For every ROS 2 terminal:

```bash
cd ~/llm_arm_ws
source ./setup_ros2_terminal.sh
```

---

# External Resources

The following resources are intentionally obtained separately rather than stored directly in this repository:

| Resource | Location |
|---|---|
| Isaac Sim scene | GitHub Releases |
| AgileX Piper description | AgileX `piper_isaac_sim` repository |
| TRAC-IK | TRACLabs `trac_ik` repository |
| Qwen2.5 model | Ollama |
| Piper voice model | Piper voice resources |
| Synthetic training dataset | Kaggle |

---

# Acknowledgements and Third-Party Software

This project uses and builds upon several open-source and research software projects:

- **ROS 2** — https://www.ros.org/
- **NVIDIA Isaac Sim** — https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
- **AgileX Piper Isaac Sim** — https://github.com/agilexrobotics/piper_isaac_sim
- **MoveIt 2** — https://moveit.picknik.ai/
- **TRAC-IK** — https://bitbucket.org/traclabs/trac_ik/
- **Ultralytics YOLO** — https://github.com/ultralytics/ultralytics
- **OpenAI Whisper** — https://github.com/openai/whisper
- **Ollama** — https://github.com/ollama/ollama
- **Qwen2.5** — https://github.com/QwenLM/Qwen2.5
- **Piper TTS** — https://github.com/OHF-Voice/piper1-gpl

Please refer to the respective projects for their licenses, model licenses, and usage terms.

---

# Project Status

The repository contains the simulation framework, ROS 2 packages, trained perception model, configuration, training resources, and optional evaluation utilities required to reproduce the modular robotic manipulation pipeline.

The Isaac Sim scene is distributed separately through GitHub Releases because of its size.

The project currently targets **simulation in NVIDIA Isaac Sim** and has not been validated as a hardware deployment.

---

## Author

**Sentoor Kumar**

M.Sc. Artificial Intelligence and Robotics  
Hof University of Applied Sciences, Germany
