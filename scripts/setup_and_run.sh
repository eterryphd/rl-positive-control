#!/bin/bash
# setup_and_run.sh
# 
# FIRST-RUN setup for GRPO training on Runpod with 4x A40
# 
# This script:
#   1. Sets up cache directories on /workspace (not root fs)
#   2. Installs pinned compatible dependencies
#   3. Verifies HuggingFace authentication (supports HF_TOKEN env var)
#   4. Validates GPU setup
#   5. Prints launch instructions
#
# For automatic restarts (spot pods), use startup.sh instead.
#
# Usage:
#   bash setup_and_run.sh
#
# With HF_TOKEN (recommended for spot pods):
#   HF_TOKEN=hf_xxxx bash setup_and_run.sh

set -e

echo "========================================"
echo "GRPO Training Setup - 4x A40 (First Run)"
echo "========================================"
echo "Time: $(date)"
echo ""

# ============================================================================
# INSTALL SYSTEM TOOLS
# ============================================================================

echo ">>> Installing system tools..."
apt-get update && apt-get install -y tmux nano
echo "✓ Installed tmux, nano"

WORKSPACE="/workspace"
REPO_NAME="rl-positive-control"
VENV_DIR="$WORKSPACE/venv"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# ============================================================================
# CACHE DIRECTORIES - Prevent filling root filesystem
# ============================================================================

echo ">>> Setting up cache directories on /workspace..."
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR"

# Also clear any existing root cache to free space
if [ -d ~/.cache/huggingface ]; then
    echo "    Clearing old HF cache from root..."
    rm -rf ~/.cache/huggingface
fi
if [ -d ~/.cache/pip ]; then
    echo "    Clearing old pip cache from root..."
    rm -rf ~/.cache/pip
fi

echo "✓ Caches configured:"
echo "    HF_HOME=$HF_HOME"
echo "    PIP_CACHE_DIR=$PIP_CACHE_DIR"

# ============================================================================
# VIRTUAL ENVIRONMENT - Persists on network volume across restarts
# ============================================================================

echo ""
echo ">>> Setting up virtual environment at $VENV_DIR..."

if [ -d "$VENV_DIR" ]; then
    echo "    Existing venv found, activating..."
    source "$VENV_DIR/bin/activate"
    echo "✓ Activated existing venv"
    
    # Check if packages are installed
    if python -c "import trl" 2>/dev/null; then
        echo "✓ Packages already installed, skipping pip install"
        SKIP_INSTALL=1
    else
        echo "    Packages missing, will reinstall..."
        SKIP_INSTALL=0
    fi
else
    echo "    Creating new venv..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "✓ Created and activated new venv"
    SKIP_INSTALL=0
fi

# ============================================================================
# INSTALL DEPENDENCIES WITH PINNED VERSIONS
# ============================================================================

if [ "$SKIP_INSTALL" != "1" ]; then
    echo ""
    echo ">>> Installing dependencies with compatible versions..."
    
    # Core packages with known-compatible versions (HF Cookbook Dec 2025)
    pip install --upgrade pip
    
    # Install in order to avoid conflicts
    # TESTED STACK: trl==0.23.1, vllm==0.11.0, transformers==4.57.0
    pip install "torch>=2.5.0"
    pip install transformers==4.57.0
    pip install "accelerate>=1.4.0"
    pip install "deepspeed>=0.16.0"
    pip install trl==0.23.1
    pip install vllm==0.11.0
    pip install datasets huggingface-hub tqdm bitsandbytes
fi

echo ""
echo ">>> Installed versions:"
pip show trl | grep -E "^(Name|Version)"
pip show accelerate | grep -E "^(Name|Version)"
pip show deepspeed | grep -E "^(Name|Version)"
pip show transformers | grep -E "^(Name|Version)"
pip show vllm | grep -E "^(Name|Version)"

# ============================================================================
# VERIFY GPU SETUP
# ============================================================================

echo ""
echo ">>> Verifying GPU setup..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
"

# ============================================================================
# HUGGINGFACE LOGIN CHECK (supports HF_TOKEN env var + network volume)
# ============================================================================

echo ""
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
    # Save to network volume for persistence
    echo -n "$HF_TOKEN" > "$WORKSPACE/.cache/huggingface/token"
    echo "  Token saved to network volume"
elif huggingface-cli whoami &>/dev/null; then
    echo "✓ Already logged in to HuggingFace"
    # Save token to network volume if not there
    if [[ -f ~/.cache/huggingface/token ]] && [[ ! -f "$WORKSPACE/.cache/huggingface/token" ]]; then
        cp ~/.cache/huggingface/token "$WORKSPACE/.cache/huggingface/token"
        echo "  Saved token to network volume for future restarts"
    fi
else
    echo "Not logged in and no HF_TOKEN set."
    echo ""
    echo "Options:"
    echo "  1. Set HF_TOKEN environment variable"
    echo "  2. Run interactive login now (will be saved to network volume)"
    echo ""
    read -p "Run interactive login? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
        # Save to network volume after login
        if [[ -f ~/.cache/huggingface/token ]]; then
            cp ~/.cache/huggingface/token "$WORKSPACE/.cache/huggingface/token"
            echo "  Token saved to network volume for future restarts"
        fi
    else
        echo "Skipping login. You'll need to set HF_TOKEN before training."
    fi
fi

# ============================================================================
# SETUP COMPLETE - PRINT LAUNCH INSTRUCTIONS
# ============================================================================

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "OPTION 1: Use the launcher script (recommended)"
echo "  cd $WORKSPACE/$REPO_NAME"
echo "  bash scripts/launch_training.sh"
echo ""
echo "OPTION 2: Manual two-terminal setup with tmux"
echo "  tmux new -s training"
echo "  # Split: Ctrl+b %"
echo ""
echo "  TERMINAL 1 (vLLM server on GPU 3):"
echo "    CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model $MODEL"
echo ""
echo "  TERMINAL 2 (Training on GPUs 0-2):"
echo "    cd $WORKSPACE/$REPO_NAME"
echo "    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \\"
echo "        --config_file scripts/accelerate_config.yaml \\"
echo "        --num_processes 3 \\"
echo "        scripts/train.py --model $MODEL"
echo ""
echo "OPTION 3: Run without vLLM (slower but simpler)"
echo "  accelerate launch --config_file scripts/accelerate_config.yaml \\"
echo "      --num_processes 4 scripts/train.py --model $MODEL --no-vllm"
echo ""
echo "========================================"
echo "FOR SPOT PODS (auto-restart)"
echo "========================================"
echo ""
echo "1. Set HF_TOKEN in Runpod pod environment variables"
echo "2. Use this Docker command:"
echo "   bash -c 'cd /workspace/$REPO_NAME && bash scripts/startup.sh'"
echo ""
echo "startup.sh will:"
echo "  - Use HF_TOKEN automatically"
echo "  - Detect and resume from latest checkpoint"
echo "  - Handle graceful shutdown on preemption"
echo ""