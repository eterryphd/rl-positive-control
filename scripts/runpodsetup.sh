#!/bin/bash
# runpod_setup.sh
# 
# Setup script for RL positive control experiments on Runpod
# Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Usage:
#   1. Start Runpod instance with the above image
#   2. SSH into the instance
#   3. Run: bash runpod_setup.sh
#
# After setup, run validation:
#   python scripts/validate_local_vs_api.py --model meta-llama/Llama-3.2-3B-Instruct --n 30

set -e  # Exit on error

echo "========================================"
echo "RL Positive Control - Runpod Setup"
echo "========================================"

# ============================================================================
# CONFIG
# ============================================================================

REPO_URL="https://github.com/eterryphd/rl-positive-control.git"
WORKSPACE="/workspace"
REPO_NAME="rl-positive-control"

# ============================================================================
# CLONE REPO
# ============================================================================

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
pip install -q transformers huggingface-hub tqdm

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
echo "Next steps:"
echo "  1. Validate local inference:"
echo "     python scripts/validate_local_vs_api.py --model meta-llama/Llama-3.2-3B-Instruct --n 30"
echo ""
echo "  2. Run evaluation:"
echo "     python scripts/evaluate.py --model meta-llama/Llama-3.2-3B-Instruct"
echo ""
echo "  3. (Coming soon) Train:"
echo "     python scripts/train.py --model meta-llama/Llama-3.2-3B-Instruct"
echo ""