#!/bin/bash
# startup.sh
#
# Runpod pod startup script - runs automatically when pod starts.
#
# To use: In Runpod pod creation, set "Docker Command" to:
#   bash -c "curl -sL https://raw.githubusercontent.com/eterryphd/rl-positive-control/main/startup.sh | bash"
#
# Or copy this entire script into the "Docker Command" field.
#
# Behavior:
#   - Always: Setup environment (clone/pull, install deps)
#   - If checkpoints exist: Resume training automatically
#   - If no checkpoints: Wait for manual intervention (first run)

set -e

WORKSPACE="/workspace"
REPO_NAME="rl-positive-control"
REPO_URL="https://github.com/eterryphd/rl-positive-control.git"
CHECKPOINT_DIR="/workspace/checkpoints"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "========================================"
echo "RL Positive Control - Pod Startup"
echo "========================================"
echo "$(date)"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo ""
echo ">>> Installing system packages..."
apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1
echo "✓ tmux installed"

echo ""
echo ">>> Setting up repository..."
cd $WORKSPACE

if [ -d "$REPO_NAME" ]; then
    echo "    Pulling latest..."
    cd $REPO_NAME
    git pull
else
    echo "    Cloning..."
    git clone $REPO_URL
    cd $REPO_NAME
fi
echo "✓ Repository ready"

echo ""
echo ">>> Installing Python dependencies..."
pip install -q transformers huggingface-hub tqdm trl datasets accelerate
echo "✓ Dependencies installed"

echo ""
echo ">>> Environment check:"
python -c "import torch; print(f'    GPUs: {torch.cuda.device_count()}')"
python -c "import torch; print(f'    CUDA: {torch.cuda.is_available()}')"

# ============================================================================
# TRAINING DECISION
# ============================================================================

echo ""
echo ">>> Checking for existing checkpoints..."

if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
    echo "✓ Checkpoints found - resuming training automatically"
    echo ""
    echo "========================================"
    echo "AUTO-RESUMING TRAINING"
    echo "========================================"
    
    cd $WORKSPACE/$REPO_NAME
    accelerate launch --multi_gpu --num_processes 4 scripts/train.py --model $MODEL
    
    echo ""
    echo "Training complete. Running final evaluation..."
    python scripts/evaluate.py --model /workspace/checkpoints/final
    
else
    echo "✗ No checkpoints found - waiting for manual start"
    echo ""
    echo "========================================"
    echo "FIRST RUN - MANUAL MODE"
    echo "========================================"
    echo ""
    echo "This appears to be a fresh start. Run these commands manually:"
    echo ""
    echo "  1. tmux"
    echo ""
    echo "  2. Baseline evaluation:"
    echo "     python scripts/evaluate.py --model $MODEL"
    echo ""
    echo "  3. Start training:"
    echo "     accelerate launch --multi_gpu --num_processes 4 scripts/train.py --model $MODEL"
    echo ""
    echo "Once training starts, if preempted, the pod will auto-resume on restart."
    echo ""
    
    # Keep pod alive for SSH access
    sleep infinity
fi