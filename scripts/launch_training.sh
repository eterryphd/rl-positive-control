#!/bin/bash
# launch_training.sh
#
# One-command launcher for GRPO training with vLLM
# 
# OPTIMIZATIONS:
#   1. Parallel model loading - vLLM and DeepSpeed init simultaneously
#   2. Local compile cache - fast I/O during torch.compile, synced to network volume
#   3. --enforce-eager on vLLM - skip CUDA graph compilation
#   4. Extended NCCL timeout - prevent false deadlock detection during compile
#
# For spot pods with auto-restart, use startup.sh instead.
#
# Usage:
#   bash launch_training.sh [--model MODEL] [--no-vllm]
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
MAX_MODEL_LENGTH=8192

# ============================================================================
# CACHE DIRECTORIES - HF/pip on network volume (large, sequential reads OK)
# ============================================================================

export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

# ============================================================================
# COMPILE CACHE - Local storage for speed, sync to network for persistence
# ============================================================================

LOCAL_CACHE="/tmp/compile_cache"
NETWORK_CACHE="$WORKSPACE/.cache/compile"

mkdir -p "$LOCAL_CACHE/torch" "$LOCAL_CACHE/triton"
mkdir -p "$NETWORK_CACHE/torch" "$NETWORK_CACHE/triton"

# Restore from network volume if exists (fast startup after first run)
if [ -d "$NETWORK_CACHE/torch" ] && [ "$(ls -A $NETWORK_CACHE/torch 2>/dev/null)" ]; then
    echo ">>> Restoring torch compile cache from network volume..."
    cp -r "$NETWORK_CACHE/torch/"* "$LOCAL_CACHE/torch/" 2>/dev/null || true
fi
if [ -d "$NETWORK_CACHE/triton" ] && [ "$(ls -A $NETWORK_CACHE/triton 2>/dev/null)" ]; then
    echo ">>> Restoring triton cache from network volume..."
    cp -r "$NETWORK_CACHE/triton/"* "$LOCAL_CACHE/triton/" 2>/dev/null || true
fi

# Point compilers at local storage
export TORCH_COMPILE_CACHE_DIR="$LOCAL_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE/torch"
export TRITON_CACHE_DIR="$LOCAL_CACHE/triton"

echo "✓ Compile caches: $LOCAL_CACHE (local) ↔ $NETWORK_CACHE (persistent)"

# ============================================================================
# NCCL CONFIG - Fix mixed P2P/SHM topology issues across NUMA nodes
# ============================================================================

export NCCL_P2P_DISABLE=1  # Force consistent SHM communication
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 min (default is 480s)
export NCCL_TIMEOUT=1800

# ============================================================================
# DISABLE TORCH COMPILE - Skip compilation overhead entirely
# ============================================================================

export TORCH_COMPILE_DISABLE=1  # Global kill switch for torch.compile

# ============================================================================
# DEBUG LOGGING - Verbose output to diagnose issues
# ============================================================================

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

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
echo "GRPO Training Launch (Optimized)"
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

# ============================================================================
# CLEANUP - Save compile cache on exit (normal or error)
# ============================================================================

cleanup() {
    local exit_code=$?
    echo ""
    echo ">>> Saving compile caches to network volume..."
    cp -r "$LOCAL_CACHE/torch/"* "$NETWORK_CACHE/torch/" 2>/dev/null || true
    cp -r "$LOCAL_CACHE/triton/"* "$NETWORK_CACHE/triton/" 2>/dev/null || true
    echo "✓ Caches saved"
    
    if [[ -n "$VLLM_PID" ]]; then
        echo ">>> Shutting down vLLM server..."
        kill $VLLM_PID 2>/dev/null || true
    fi
    
    exit $exit_code
}
trap cleanup EXIT

# ============================================================================
# MODEL CACHE CHECK (skip download if already cached)
# ============================================================================

MODEL_CACHE_NAME="${MODEL//\//-}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--$MODEL_CACHE_NAME"

if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✓ Model already cached: $MODEL_CACHE_DIR"
else
    echo ">>> Downloading model (first run only): $MODEL"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL', cache_dir='$HF_HOME')"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download model. Check network, HF_TOKEN, and model access."
        exit 1
    fi
fi
echo ""

# ============================================================================
# PARALLEL STARTUP: vLLM + Training load models simultaneously
# ============================================================================

if $USE_VLLM; then
    echo ">>> Starting vLLM server on GPU 3 (background)..."
    echo "    --enforce-eager: skip CUDA graph compilation for faster startup"
    CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
        --model "$MODEL" \
        --port $VLLM_PORT \
        --max-model-len $MAX_MODEL_LENGTH \
        --enforce-eager &
    VLLM_PID=$!
    
    # NO WAIT HERE - training starts immediately
    # train.py will wait for vLLM after DeepSpeed init completes
    
    echo ">>> Launching training on GPUs 0-2 (parallel with vLLM startup)..."
    echo "    Training will wait for vLLM after model initialization"
    echo ""
    
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