#!/usr/bin/env python3
"""
RL training for arithmetic task using GRPO.

This is a positive control to validate the RL pipeline before
applying to more complex tasks like interleaving.

TESTED STACK (HuggingFace Cookbook, Dec 2025):
    trl==0.23.1, vllm==0.11.0, transformers==4.57.0

OPTIMIZATIONS:
    - Waits for vLLM AFTER DeepSpeed init (parallel model loading)
    - Saves 6-8 minutes of startup time

Usage (with vLLM server mode - recommended for 4x A40):
    # Terminal 1: Start vLLM server on GPU 3
    CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model meta-llama/Llama-3.1-8B-Instruct
    
    # Terminal 2: Launch training on GPUs 0,1,2
    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes 3 \
        train.py --model meta-llama/Llama-3.1-8B-Instruct

Usage (without vLLM - slower but simpler):
    accelerate launch --config_file accelerate_config.yaml train.py \
        --model meta-llama/Llama-3.1-8B-Instruct --no-vllm
"""

import argparse
import json
import math
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

# Set cache directories before importing HF libraries
# This prevents filling up the root filesystem on Runpod
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
if 'PIP_CACHE_DIR' not in os.environ:
    os.environ['PIP_CACHE_DIR'] = '/workspace/.cache/pip'

from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import trl

# ============================================================================
# CONFIG - User configurable settings
# ============================================================================

CONFIG = {
    # Prompt configuration - MUST MATCH evaluate.py
    'system_message': "You are a calculator. Output only the number.",
    'max_new_tokens': 4096,  # Interleaved output needs ~1000 tokens + buffer
    'max_prompt_length': 2048,  # Full texts + instructions ~1200 tokens
    
    # Generation
    'temperature': 0.9,  # More diversity for reward variance
    'num_generations': 6,  # More samples for GRPO signal
    
    # Training - tuned for 4x A40 (48GB each)
    'learning_rate': 1e-6,
    'num_train_epochs': 1,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'max_steps': 200,
    'logging_steps': 10,
    # NOTE: save_steps set high - RewardThresholdSaveCallback handles saving
    # based on reward threshold (>0.3) or new best
    'save_steps': 999999,
    
    # GRPO specific
    'beta': 0.0,  # KL penalty - 0.0 is now standard (no ref model needed)
    
    # Paths
    'data_dir': 'data',
    'output_dir': '/workspace/checkpoints',
}

# ============================================================================
# VLLM SERVER WAIT (for parallel loading optimization)
# ============================================================================

def wait_for_vllm(host: str, port: int, timeout: int = 1800, check_interval: int = 5):
    """
    Wait for vLLM server to be ready.
    
    Called AFTER DeepSpeed init to allow parallel model loading.
    This saves 6-8 minutes compared to waiting before training starts.
    
    Args:
        host: vLLM server host
        port: vLLM server port
        timeout: Maximum seconds to wait (default 30 min to handle slow compiles)
        check_interval: Seconds between health checks
    
    Returns:
        True if server is ready
        
    Raises:
        RuntimeError if server not ready within timeout
    """
    url = f"http://{host}:{port}/health"
    start = time.time()
    last_print = 0
    
    print(f">>> Waiting for vLLM server at {host}:{port}...")
    
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            elapsed = time.time() - start
            print(f"✓ vLLM server ready (waited {elapsed:.0f}s)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            elapsed = time.time() - start
            # Print progress every 30 seconds
            if elapsed - last_print >= 30:
                print(f"    Still waiting for vLLM... ({elapsed:.0f}s)")
                last_print = elapsed
            time.sleep(check_interval)
    
    raise RuntimeError(f"vLLM server not ready after {timeout}s")


# ============================================================================
# REWARD FUNCTION
# ============================================================================

def extract_answer(response: str):
    """Extract numeric answer from model response."""
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    Compute rewards for completions using continuous relative error.
    
    Reward based on how close the answer is - exponential decay.
    Perfect = +1.0, order of magnitude off ≈ -0.3, way off → -1.0
    
    Args:
        prompts: List of prompt strings
        completions: List of completion strings (model outputs)
        **kwargs: May contain 'answer' from dataset
    
    Returns:
        List of reward floats in [-1, 1]
    """
    rewards = []
    answers = kwargs.get('answer', [None] * len(prompts))
    
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        
        if predicted is None or answer is None:
            rewards.append(-1.0)
            continue
        
        # Relative error (add 1 to denominator to handle answer=0)
        relative_error = abs(predicted - answer) / (abs(answer) + 1)
        
        # Exponential decay: perfect=1.0, 10% off≈0.6, 50% off≈0.08
        raw_reward = math.exp(-relative_error * 5)
        
        # Scale to [-1, 1]
        reward = 2 * raw_reward - 1
        
        rewards.append(reward)
    
    return rewards


# ============================================================================
# DATA LOADING
# ============================================================================

def build_prompt(problem: str, tokenizer) -> str:
    """Build chat-formatted prompt."""
    messages = [
        {"role": "system", "content": CONFIG['system_message']},
        {"role": "user", "content": problem}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def load_dataset_for_grpo(tokenizer) -> Dataset:
    """Load and format dataset for GRPO trainer."""
    data_path = Path(CONFIG['data_dir']) / 'train.json'
    
    with open(data_path) as f:
        problems = json.load(f)
    
    # GRPO expects 'prompt' column
    data = {
        'prompt': [build_prompt(p['problem'], tokenizer) for p in problems],
        'problem': [p['problem'] for p in problems],
        'answer': [p['answer'] for p in problems],
    }
    
    return Dataset.from_dict(data)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def get_checkpoints(output_dir: Path) -> list:
    """Get sorted list of checkpoint directories (by step number)."""
    checkpoints = []
    if not output_dir.exists():
        return checkpoints
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith('checkpoint-'):
            try:
                step = int(d.name.split('-')[1])
                checkpoints.append((step, d))
            except (IndexError, ValueError):
                continue
    return sorted(checkpoints, key=lambda x: x[0])


def cleanup_checkpoints(output_dir: Path):
    """Keep only: first, second-to-last, and last valid checkpoint. Remove invalid ones."""
    checkpoints = get_checkpoints(output_dir)
    
    # First, remove any invalid checkpoints
    valid_checkpoints = []
    for step, path in checkpoints:
        if validate_checkpoint(path):
            valid_checkpoints.append((step, path))
        else:
            print(f"    Removing invalid checkpoint: {path.name}")
            shutil.rmtree(path)
    
    if len(valid_checkpoints) <= 3:
        return
    
    keep_indices = {0, len(valid_checkpoints) - 2, len(valid_checkpoints) - 1}
    
    for i, (step, path) in enumerate(valid_checkpoints):
        if i not in keep_indices:
            print(f"    Removing old checkpoint: {path.name}")
            shutil.rmtree(path)


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """Check if checkpoint has all required files for resumption."""
    required_files = [
        'trainer_state.json',  # Training state
    ]
    
    for f in required_files:
        if not (checkpoint_path / f).exists():
            return False
    
    # Check for at least one model/optimizer shard
    has_model_files = any(
        checkpoint_path.glob('*.safetensors')
    ) or any(
        checkpoint_path.glob('*.bin')
    ) or (checkpoint_path / 'pytorch_model.bin').exists()
    
    # For DeepSpeed, check for ZeRO checkpoint
    has_deepspeed = (checkpoint_path / 'zero_to_fp32.py').exists() or \
                    any(checkpoint_path.glob('global_step*'))
    
    return has_model_files or has_deepspeed


def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Find latest valid checkpoint for resumption."""
    checkpoints = get_checkpoints(output_dir)
    
    # Try checkpoints from newest to oldest
    for step, path in reversed(checkpoints):
        if validate_checkpoint(path):
            return str(path)
        else:
            print(f"    Warning: Checkpoint {path.name} is incomplete, skipping...")
    
    return None


class CheckpointCleanupCallback(TrainerCallback):
    """Clean up old checkpoints after each save. Only runs on main process."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved."""
        # Only run cleanup on main process to avoid race conditions
        if state.is_world_process_zero:
            cleanup_checkpoints(self.output_dir)


class RewardThresholdSaveCallback(TrainerCallback):
    """
    Only save checkpoints when reward exceeds threshold or is new best.
    
    This prevents filling disk with mediocre checkpoints while ensuring
    we capture good training states.
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.best_reward = float('-inf')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        reward = logs.get('reward', None)
        if reward is None:
            return
        
        # Save if above threshold OR new best
        if reward > self.threshold or reward > self.best_reward:
            if reward > self.best_reward:
                self.best_reward = reward
                print(f">>> New best reward: {reward:.3f} - saving checkpoint")
            else:
                print(f">>> Reward {reward:.3f} > {self.threshold} threshold - saving checkpoint")
            control.should_save = True


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function."""
    print("=" * 70)
    print("GRPO TRAINING - ARITHMETIC POSITIVE CONTROL")
    print("=" * 70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for GRPO
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = load_dataset_for_grpo(tokenizer)
    print(f"    Train: {len(train_dataset)} examples")
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint to resume from
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from:
        print(f"\n>>> Found existing checkpoint: {resume_from}")
        print("    Will resume training from this checkpoint.")
    
    # GRPO config - with proper multi-GPU settings
    print("\nConfiguring GRPO...")
    
    grpo_kwargs = {
        'output_dir': str(output_dir),
        'learning_rate': CONFIG['learning_rate'],
        'num_train_epochs': CONFIG['num_train_epochs'],
        'per_device_train_batch_size': CONFIG['per_device_train_batch_size'],
        'gradient_accumulation_steps': CONFIG['gradient_accumulation_steps'],
        'num_generations': CONFIG['num_generations'],
        'max_completion_length': CONFIG['max_new_tokens'],
        'max_prompt_length': CONFIG['max_prompt_length'],
        'temperature': CONFIG['temperature'],  # Diversity for reward variance
        'max_steps': CONFIG['max_steps'],
        'logging_steps': CONFIG['logging_steps'],
        'save_steps': CONFIG['save_steps'],
        'lr_scheduler_type': 'constant',  # No decay - let Adam handle adaptation
        # Keeping all checkpoints - identify best by validation later
        # NOTE: save_only_model=False required for DeepSpeed resume on spot pods
        # Checkpoints are larger but can actually resume training
        'save_only_model': False,
        'beta': CONFIG['beta'],
        'bf16': True,
        'optim': 'adamw_bnb_8bit',  # 8-bit Adam - half optimizer memory
        'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},  # Fix for shape mismatch
        'remove_unused_columns': False,  # Keep 'answer' column for reward fn
        'report_to': 'none',  # Disable wandb unless you want it
    }
    
    # Add vLLM config if enabled
    if args.use_vllm:
        print("    vLLM: ENABLED (server mode)")
        print(f"    TRL version: {trl.__version__}")
        
        # TRL API changed over versions - detect what's available
        import inspect
        grpo_params = inspect.signature(GRPOConfig).parameters
        
        grpo_kwargs['use_vllm'] = True
        
        # vllm_mode was added in TRL ~0.16+
        if 'vllm_mode' in grpo_params:
            grpo_kwargs['vllm_mode'] = 'server'
            print("    Using vllm_mode='server'")
        
        # Server host/port config
        if 'vllm_server_host' in grpo_params:
            grpo_kwargs['vllm_server_host'] = args.vllm_host
            grpo_kwargs['vllm_server_port'] = args.vllm_port
        elif 'vllm_server_url' in grpo_params:
            # Some versions use URL instead
            grpo_kwargs['vllm_server_url'] = f"http://{args.vllm_host}:{args.vllm_port}"
        
        print(f"    Server: {args.vllm_host}:{args.vllm_port}")
    else:
        print("    vLLM: DISABLED (using transformers generation)")
    
    training_args = GRPOConfig(**grpo_kwargs)
    
    # Initialize trainer (this triggers DeepSpeed ZeRO-3 init)
    print("\nInitializing GRPO trainer (DeepSpeed ZeRO-3 sharding)...")
    init_start = time.time()
    
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Add reward-based checkpoint saving
    trainer.add_callback(RewardThresholdSaveCallback(threshold=0.3))
    
    init_elapsed = time.time() - init_start
    print(f"✓ Trainer initialized ({init_elapsed:.0f}s)")
    
    # NOW wait for vLLM (after DeepSpeed init, so both load in parallel)
    if args.use_vllm:
        wait_for_vllm(args.vllm_host, args.vllm_port)
    
    # Train
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Keeping all checkpoints for analysis
    # cleanup_checkpoints(output_dir)
    
    # Training complete - final checkpoint is already saved by trainer
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    remaining = get_checkpoints(output_dir)
    print(f"\nCheckpoints retained: {[c[1].name for c in remaining]}")
    print(f"Use the latest checkpoint for evaluation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name')
    parser.add_argument('--use-vllm', action='store_true', default=True,
                        help='Use vLLM for generation (default: True)')
    parser.add_argument('--no-vllm', action='store_false', dest='use_vllm',
                        help='Disable vLLM (use transformers generation)')
    parser.add_argument('--vllm-host', type=str, default='localhost',
                        help='vLLM server host')
    parser.add_argument('--vllm-port', type=int, default=8000,
                        help='vLLM server port')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()