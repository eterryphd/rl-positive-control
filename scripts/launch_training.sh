#!/bin/bash
# launch_training.sh
# Script to launch GRPO training with optimized cache handling

# ==============================================================================
# CONFIG
# ==============================================================================
# Cache directories - use local SSD for speed, network volume for persistence
LOCAL_CACHE_DIR="/tmp/cache"
NETWORK_CACHE_DIR="/workspace/cache"

# Triton/Torch compile caches - these are large and frequently accessed
TRITON_CACHE_DIR="${LOCAL_CACHE_DIR}/triton"
TORCH_CACHE_DIR="${LOCAL_CACHE_DIR}/torch"

# Timeouts - prevent deadlocks
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_COUNT=20

# ==============================================================================
# CACHE MANAGEMENT
# ==============================================================================

# Function to sync cache from network to local
sync_cache() {
    if [ -d "$NETWORK_CACHE_DIR" ]; then
        echo ">>> Syncing cache from network volume..."
        cp -r "$NETWORK_CACHE_DIR/"* "$LOCAL_CACHE_DIR/"  # Use cp if rsync unavailable
    else
        echo ">>> No network cache found, using fresh local cache"
    fi
}

# Function to validate cache integrity
validate_cache() {
    echo ">>> Validating cache integrity..."
    
    # Check for common corruption patterns
    find "$LOCAL_CACHE_DIR" -type f \( -name "*.pickle" -o -name "*.json" \) -exec bash -c 'if ! python3 -c "import pickle; pickle.load(open(\"$0\", \"rb\"))" 2>/dev/null; then echo "Corrupt file: $0"; rm "$0"; fi' {} \; || true
    
    # If cache is empty or corrupt, we'll let compiles happen naturally
}

# Function to sync back to network (on exit)
sync_back() {
    echo ">>> Syncing cache back to network volume..."
    cp -r "$LOCAL_CACHE_DIR/"* "$NETWORK_CACHE_DIR/"  # Use cp if rsync unavailable
}

# Trap for cleanup
trap sync_back EXIT

# Create directories
mkdir -p "$LOCAL_CACHE_DIR" "$NETWORK_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_CACHE_DIR"

# Export cache env vars
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export TORCH_COMPILE_CACHE="$TORCH_CACHE_DIR"

# Sync and validate
sync_cache
validate_cache

# ==============================================================================
# HF TOKEN CHECK
# ==============================================================================
if [ -z "$HF_TOKEN" ]; then
    echo "✗ HF_TOKEN not set"
    echo "  Set environment variable: export HF_TOKEN=hf_..."
    exit 1
fi

# ==============================================================================
# VLLM SERVER
# ==============================================================================
VLLM_HOST="0.0.0.0"
VLLM_PORT=8000

if [ "$1" != "--no-vllm" ]; then
    echo ">>> Starting vLLM server on GPU 3..."
    CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 &
    VLLM_PID=$!
fi

# ==============================================================================
# MODEL DOWNLOAD (if not cached)
# ==============================================================================
echo ">>> Checking/downloading model..."
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', token='$HF_TOKEN')"

# ==============================================================================
# LAUNCH TRAINING
# ==============================================================================
echo ">>> Launching training..."

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 3 \
    train.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dynamic_data  # Enable new dynamic data approach

# Wait for training to complete
wait

# If vLLM was started, clean up
if [ -n "$VLLM_PID" ]; then
    kill $VLLM_PID
fi