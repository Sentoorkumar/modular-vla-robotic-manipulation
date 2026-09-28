#!/bin/bash

# ============================================================
# Modular VLA Robotic Manipulation System
# Model Setup Script
# ============================================================
#
# Prepares the AI models used by the project:
#   - Qwen2.5 7B Instruct Q4_0 via Ollama
#   - OpenAI Whisper small for STT
#   - Piper en_US-lessac-medium for TTS
#   - Custom YOLOv8 best.pt verification
#
# Tested project configuration:
#   Ubuntu 22.04 / Python 3.10
# ============================================================

set -e

WORKSPACE="$HOME/llm_arm_ws"
PIPER_VOICE_DIR="$HOME/.local/share/piper-voices"

echo "============================================================"
echo " Modular VLA - Model Setup"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. System dependency check
# ------------------------------------------------------------

echo "[1/5] Checking FFmpeg..."

if command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg found."
else
    echo "FFmpeg not found. Installing..."
    sudo apt update
    sudo apt install -y ffmpeg
fi

# ------------------------------------------------------------
# 2. Python AI packages
# ------------------------------------------------------------

echo
echo "[2/5] Installing Python model packages..."

python3 -m pip install --user \
    openai-whisper==20250625 \
    piper-tts==1.4.1 \
    ultralytics==8.3.222

# ------------------------------------------------------------
# 3. Whisper model
# ------------------------------------------------------------

echo
echo "[3/5] Preparing Whisper small model..."

python3 - <<'PY'
import whisper

print("Loading/downloading Whisper small model...")
whisper.load_model("small")
print("Whisper small model ready.")
PY

# ------------------------------------------------------------
# 4. Piper TTS voice
# ------------------------------------------------------------

echo
echo "[4/5] Preparing Piper TTS voice..."

mkdir -p "$PIPER_VOICE_DIR"

PIPER_MODEL="$PIPER_VOICE_DIR/en_US-lessac-medium.onnx"
PIPER_CONFIG="$PIPER_VOICE_DIR/en_US-lessac-medium.onnx.json"

PIPER_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

if [ ! -f "$PIPER_MODEL" ]; then
    echo "Downloading Piper voice model..."
    wget -O "$PIPER_MODEL" \
        "$PIPER_BASE_URL/en_US-lessac-medium.onnx"
else
    echo "Piper voice model already exists."
fi

if [ ! -f "$PIPER_CONFIG" ]; then
    echo "Downloading Piper voice configuration..."
    wget -O "$PIPER_CONFIG" \
        "$PIPER_BASE_URL/en_US-lessac-medium.onnx.json"
else
    echo "Piper voice configuration already exists."
fi

# ------------------------------------------------------------
# 5. Ollama / Qwen and YOLO verification
# ------------------------------------------------------------

echo
echo "[5/5] Preparing Qwen LLM and checking YOLO model..."

if ! command -v ollama >/dev/null 2>&1; then
    echo
    echo "ERROR: Ollama is not installed."
    echo "Install Ollama first, then run this script again."
    echo
    exit 1
fi

echo "Checking Qwen2.5 model..."

if ollama list | grep -q "qwen2.5:7b-instruct-q4_0"; then
    echo "Qwen2.5 model already exists."
else
    echo "Downloading Qwen2.5 7B Instruct Q4_0..."
    ollama pull qwen2.5:7b-instruct-q4_0
fi

YOLO_MODEL="$WORKSPACE/src/perception_agent/models/best.pt"

if [ -f "$YOLO_MODEL" ]; then
    echo "Custom YOLO model found:"
    echo "  $YOLO_MODEL"
else
    echo
    echo "WARNING: Custom YOLO model was not found:"
    echo "  $YOLO_MODEL"
fi

echo
echo "============================================================"
echo " Model setup completed."
echo "============================================================"
echo
echo "Configured models:"
echo "  STT   : Whisper small"
echo "  LLM   : qwen2.5:7b-instruct-q4_0"
echo "  TTS   : Piper en_US-lessac-medium"
echo "  Vision: Custom YOLO best.pt"
echo
