# GRPO Training Scripts for 4x A40

## Quick Start

### On-Demand Pod (Interactive)
```bash
cd /workspace/rl-positive-control
bash scripts/setup_and_run.sh   # First time only - creates venv, installs deps
bash scripts/launch_training.sh  # Start training
```

### Spot Pod (Auto-Resume)
Set in Runpod pod configuration:
- **Environment Variables:**
  - `HF_TOKEN=hf_xxxx`
  - `MODEL=meta-llama/Llama-3.1-8B-Instruct` (optional)
- **Docker Command:**
  ```bash
  bash -c "cd /workspace/rl-positive-control && bash scripts/startup.sh"
  ```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_and_run.sh` | First-time setup: creates venv on network volume, installs all deps |
| `launch_training.sh` | Manual training launcher (interactive use) |
| `startup.sh` | Non-interactive auto-start for Docker CMD / spot restarts |
| `train.py` | GRPO training script |
| `evaluate.py` | Evaluation script for checkpoints |

## Architecture

```
/workspace/                     # Network volume (persists!)
├── venv/                       # Python virtual environment
├── .cache/
│   ├── huggingface/           # Model weights cache
│   ├── pip/                   # Pip cache
│   └── triton/                # Triton cache
├── checkpoints/               # Training checkpoints
│   ├── checkpoint-10/
│   ├── checkpoint-20/
│   └── checkpoint-30/         # Only keeps last 3
└── rl-positive-control/       # This repo
    ├── scripts/
    └── data/
```

## Tested Stack

From HuggingFace Cookbook (Dec 2025):
```
trl==0.23.1
vllm==0.11.0
transformers==4.57.0
accelerate>=1.4.0
deepspeed>=0.16.0
bitsandbytes
```

## GPU Layout

- **GPUs 0-2**: Training (DeepSpeed ZeRO-3)
- **GPU 3**: vLLM server (generation)

## Key Config (train.py)

```python
'num_generations': 6,      # Generations per prompt for GRPO variance
'temperature': 0.9,        # Diversity for reward signal
'save_steps': 10,          # ~10 min between checkpoints (spot-safe)
'save_total_limit': 3,     # Keep only last 3 checkpoints
'optim': 'adamw_bnb_8bit', # 8-bit Adam for memory efficiency
```

## Troubleshooting

**OOM on GPU 3 (vLLM):**
```bash
pkill -9 -f vllm
pkill -9 -f python
# Then restart
```

**Disk full:**
```bash
rm -rf /workspace/checkpoints/checkpoint-*  # Clear old checkpoints
rm -rf /workspace/.cache/pip/*              # Clear pip cache
```

**grad_norm: 0.0 (no learning):**
- Increase `num_generations` (need reward variance)
- Increase `temperature`
- Check reward function is returning varied values