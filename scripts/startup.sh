#!/bin/bash
# startup.sh
#
# NON-INTERACTIVE startup script for Docker CMD / spot pod restarts
# Designed to be called automatically when pod starts/restarts
#
# This script:
#   - Activates venv from network volume (packages persist!)
#   - Runs setup if venv doesn't exist
#   - Resumes from latest checkpoint automatically
#
# Requirements:
#   - HF_TOKEN environment variable must be set (or already logged in)
#
# Runpod Docker command example:
#   bash -c "cd /workspace/rl-positive-control && bash scripts/startup.sh"

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
VLLM_PORT="${VLLM_PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace"
VENV_DIR="$WORKSPACE/venv"

# Cache directories - prevent filling root filesystem
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "========================================"
echo "GRPO Training - Spot Pod Startup"
echo "========================================"
echo "Time: $(date)"
echo "Model: $MODEL"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# ============================================================================
# SYSTEM TOOLS (must reinstall each restart - not on network volume)
# ============================================================================

echo ">>> Installing system tools..."
apt-get update && apt-get install -y tmux nano > /dev/null 2>&1
echo "✓ Installed tmux, nano"

# ============================================================================
# VIRTUAL ENVIRONMENT - Persists on network volume!
# ============================================================================

echo ""
echo ">>> Checking virtual environment..."

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    
    # Verify packages are there
    if python -c "import trl, vllm, deepspeed" 2>/dev/null; then
        echo "✓ Activated venv with all packages"
    else
        echo "✗ Venv exists but packages missing - running setup..."
        source "$SCRIPT_DIR/setup_and_run.sh"
    fi
else
    echo "✗ No venv found - running first-time setup..."
    source "$SCRIPT_DIR/setup_and_run.sh"
fi

# ============================================================================
# HF_TOKEN HANDLING (NON-INTERACTIVE)
# ============================================================================

echo ""
echo ">>> Checking HuggingFace authentication..."

if [[ -n "$HF_TOKEN" ]]; then
    echo "✓ Using HF_TOKEN from environment"
    # Set token for huggingface-hub
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    # Also write to cache for libraries that check there
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
elif huggingface-cli whoami &>/dev/null; then
    echo "✓ Already logged in to HuggingFace"
else
    echo "✗ FATAL: No HF_TOKEN environment variable and not logged in"
    echo ""
    echo "For spot pods, you must set HF_TOKEN in the pod configuration:"
    echo "  1. Go to Runpod pod settings"
    echo "  2. Add environment variable: HF_TOKEN=hf_xxxx"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# ============================================================================
# GPU VERIFICATION
# ============================================================================

echo ""
echo ">>> Verifying GPU setup..."
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [[ "$GPU_COUNT" -lt 4 ]]; then
    echo "✗ WARNING: Expected 4 GPUs, found $GPU_COUNT"
    echo "  Continuing anyway, but training may fail or be suboptimal"
else
    echo "✓ Found $GPU_COUNT GPUs"
fi

# ============================================================================
# CHECKPOINT DETECTION (AUTO-RESUME)
# ============================================================================

echo ""
echo ">>> Checking for existing checkpoints..."

RESUME_CHECKPOINT=""
if [[ -d "$CHECKPOINT_DIR" ]]; then
    # Find the latest checkpoint-XXXX directory
    LATEST=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    if [[ -n "$LATEST" && -d "$LATEST" ]]; then
        RESUME_CHECKPOINT="$LATEST"
        echo "✓ Found checkpoint to resume from: $(basename $RESUME_CHECKPOINT)"
        
        # List all retained checkpoints
        echo "  Retained checkpoints:"
        ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | while read ckpt; do
            echo "    - $(basename $ckpt)"
        done
    else
        echo "  No checkpoints found - starting fresh"
    fi
else
    echo "  Checkpoint directory doesn't exist - starting fresh"
    mkdir -p "$CHECKPOINT_DIR"
fi

# ============================================================================
# CLEANUP FUNCTION
# ============================================================================

cleanup() {
    echo ""
    echo ">>> Shutting down ($(date))..."
    if [[ -n "$VLLM_PID" ]]; then
        echo "  Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    echo "  Cleanup complete"
}
trap cleanup EXIT INT TERM

# ============================================================================
# START VLLM SERVER
# ============================================================================

echo ""
echo ">>> Starting vLLM server on GPU 3..."
CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model "$MODEL" --port $VLLM_PORT &
VLLM_PID=$!

# Wait for vLLM to be ready
echo ">>> Waiting for vLLM server..."
MAX_WAIT=180  # 3 minutes - model loading can take a while
WAITED=0
while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "✗ FATAL: vLLM server failed to start within ${MAX_WAIT}s"
        echo "  Check GPU memory and model availability"
        exit 1
    fi
    # Check if vLLM process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "✗ FATAL: vLLM server process died"
        exit 1
    fi
    echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
done
echo "✓ vLLM server ready"

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
echo "Time: $(date)"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "Resuming from: $(basename $RESUME_CHECKPOINT)"
else
    echo "Starting fresh (no checkpoint)"
fi
echo ""

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
    --num_processes 3 \
    "$SCRIPT_DIR/train.py" \
    --model "$MODEL" \
    --use-vllm \
    --vllm-port $VLLM_PORT

TRAIN_EXIT=$?

echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Time: $(date)"
echo "Exit code: $TRAIN_EXIT"

exit $TRAIN_EXIT