#!/bin/bash
# runpod_setup.sh
# 
# Setup script for RL positive control experiments on Runpod
# Hardware: 4x A40 (192GB total)
# Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Usage:
#   1. Start Runpod instance (4x A40, spot for cost savings)
#   2. SSH into the instance
#   3. Run: bash runpod_setup.sh
#
# Supports spot instances with automatic checkpoint resumption.

set -e  # Exit on error

echo "========================================"
echo "RL Positive Control - Runpod Setup"
echo "========================================"

# ============================================================================
# SYSTEM PACKAGES
# ============================================================================

echo ""
echo ">>> Installing system packages..."
apt-get update -qq && apt-get install -y -qq tmux
echo "✓ tmux installed (use 'tmux' to persist sessions)"

# ============================================================================
# CLONE REPO
# ============================================================================

REPO_URL="https://github.com/eterryphd/rl-positive-control.git"
WORKSPACE="/workspace"
REPO_NAME="rl-positive-control"

echo ""
echo ">>> Cloning repository..."
cd $WORKSPACE

if [ -d "$REPO_NAME" ]; then
    echo "    Repository already exists, pulling latest..."
    cd $REPO_NAME
    git pull
else
    git clone $REPO_URL
    cd $REPO_NAME
fi

echo "✓ Repository ready at $WORKSPACE/$REPO_NAME"

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

echo ""
echo ">>> Installing Python dependencies..."
pip install -q transformers huggingface-hub tqdm trl datasets accelerate

echo "✓ Dependencies installed"

# ============================================================================
# HUGGINGFACE LOGIN
# ============================================================================

echo ""
echo ">>> Checking HuggingFace authentication..."

# Check if already logged in
if huggingface-cli whoami &>/dev/null; then
    echo "✓ Already logged in to HuggingFace"
else
    echo "    Not logged in. Running huggingface-cli login..."
    echo "    You will need your HF token (from huggingface.co/settings/tokens)"
    echo ""
    huggingface-cli login
fi

# ============================================================================
# VERIFY SETUP
# ============================================================================

echo ""
echo ">>> Verifying setup..."

# Check Python packages
python -c "import transformers; import torch; print(f'    transformers: {transformers.__version__}')"
python -c "import torch; print(f'    torch: {torch.__version__}')"
python -c "import torch; print(f'    CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'    GPU count: {torch.cuda.device_count()}')"

# Check data files exist
if [ -f "data/test.json" ]; then
    echo "✓ Data files present"
else
    echo "⚠ Data files missing - run: python scripts/generate_dataset.py"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "IMPORTANT: Use tmux for long-running jobs!"
echo "  tmux                    # Start new session"
echo "  tmux attach             # Reconnect after SSH drop"
echo ""
echo "Next steps:"
echo "  1. Start tmux session:"
echo "     tmux"
echo ""
echo "  2. Run baseline evaluation (8B):"
echo "     python scripts/evaluate.py --model meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "  3. Train with multi-GPU:"
echo "     accelerate launch --multi_gpu --num_processes 4 scripts/train.py --model meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "  4. Evaluate trained model:"
echo "     python scripts/evaluate.py --model /workspace/checkpoints/final"
echo ""