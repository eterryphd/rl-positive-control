#!/bin/bash
# launch_training.sh
#
# One-command launcher for GRPO training with vLLM
# Starts vLLM server in background, waits for it, then launches training
#
# For spot pods with auto-restart, use startup.sh instead.
#
# Usage:
#   bash launch_training.sh [--model MODEL] [--no-vllm]
#
# Optional Usage:
#  bash scripts/launch_training.sh 2>&1 & | tee -a training.log
#
# With HF_TOKEN:
#   HF_TOKEN=hf_xxxx bash launch_training.sh

set -e

# Defaults
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
USE_VLLM=true
VLLM_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace"
VENV_DIR="$WORKSPACE/venv"

# Cache directories - prevent filling root filesystem
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR"

# ============================================================================
# ACTIVATE VENV
# ============================================================================

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✓ Activated venv: $VENV_DIR"
else
    echo "✗ No venv found at $VENV_DIR"
    echo "  Run setup_and_run.sh first"
    exit 1
fi

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --no-vllm) USE_VLLM=false; shift ;;
        --port) VLLM_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "GRPO Training Launch"
echo "========================================"
echo "Model: $MODEL"
echo "vLLM:  $USE_VLLM"
echo ""

# ============================================================================
# HF_TOKEN CHECK
# ============================================================================

echo ">>> Checking HuggingFace authentication..."

# Try loading from network volume first
if [[ -z "$HF_TOKEN" ]] && [[ -f "$WORKSPACE/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$WORKSPACE/.cache/huggingface/token")
    echo "  Loaded HF_TOKEN from network volume"
fi

if [[ -n "$HF_TOKEN" ]]; then
    echo "✓ Using HF_TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    mkdir -p ~/.cache/huggingface
    echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
elif huggingface-cli whoami &>/dev/null; then
    echo "✓ Already logged in to HuggingFace"
else
    echo "✗ Not authenticated with HuggingFace"
    echo "  Either set HF_TOKEN or run: huggingface-cli login"
    exit 1
fi
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "$VLLM_PID" ]]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ============================================================================
# MODEL PRE-DOWNLOAD
# ============================================================================

echo ">>> Pre-downloading model if not cached: $MODEL"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL', cache_dir='$HF_HOME')"
if [ $? -eq 0 ]; then
    echo "Model ready in cache"
else
    echo "ERROR: Failed to prepare model. Check network, HF_TOKEN, and model access."
    exit 1
fi
echo ""

# ============================================================================
# VLLM SERVER STARTUP
# ============================================================================
if $USE_VLLM; then
    echo ">>> Starting vLLM server on GPU 3..."
    CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model "$MODEL" --port $VLLM_PORT &
    VLLM_PID=$!
    
    # Wait for vLLM to be ready
    echo ">>> Waiting for vLLM server to start..."
    MAX_WAIT=900
    WAITED=0
    while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "ERROR: vLLM server failed to start within ${MAX_WAIT}s"
            exit 1
        fi
        echo "    Waiting... (${WAITED}s)"
    done
    echo "✓ vLLM server ready"
    
    # Launch training on GPUs 0-2
    echo ""
    echo ">>> Launching training on GPUs 0-2..."
    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
        --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
        --num_processes 3 \
        "$SCRIPT_DIR/train.py" \
        --model "$MODEL" \
        --use-vllm \
        --vllm-port $VLLM_PORT
else
    echo ">>> Launching training on all 4 GPUs (no vLLM)..."
    
    # Modify accelerate config for 4 GPUs
    sed 's/num_processes: 3/num_processes: 4/' "$SCRIPT_DIR/accelerate_config.yaml" > /tmp/accelerate_config_4gpu.yaml
    
    accelerate launch \
        --config_file /tmp/accelerate_config_4gpu.yaml \
        --num_processes 4 \
        "$SCRIPT_DIR/train.py" \
        --model "$MODEL" \
        --no-vllm
fi

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"