#!/usr/bin/env python3
"""
RL training for arithmetic task using GRPO.

This is a positive control to validate the RL pipeline before
applying to more complex tasks like interleaving.

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
import os
import re
import shutil
from pathlib import Path

# Set cache directories before importing HF libraries
# This prevents filling up the root filesystem on Runpod
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
if 'PIP_CACHE_DIR' not in os.environ:
    os.environ['PIP_CACHE_DIR'] = '/workspace/.cache/pip'

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import trl

# ============================================================================
# CONFIG - User configurable settings
# ============================================================================

CONFIG = {
    # Prompt configuration - MUST MATCH evaluate.py
    'system_message': "You are a calculator. Output only the number.",
    'max_new_tokens': 15,
    
    # Reward
    'reward_correct': 1.0,
    'reward_incorrect': -1.0,  # GRPO works better with negative rewards
    'tolerance': 0.01,
    
    # Training - tuned for 4x A40 (48GB each)
    'learning_rate': 1e-6,
    'num_train_epochs': 1,
    'per_device_train_batch_size': 1,  # Small for 8B model
    'gradient_accumulation_steps': 8,  # Effective batch = 1 * 3 GPUs * 8 = 24
    'num_generations': 4,  # Completions per prompt for GRPO
    'max_steps': 100,
    'logging_steps': 10,
    'save_steps': 25,
    
    # GRPO specific
    'beta': 0.0,  # KL penalty - 0.0 is now standard (no ref model needed)
    
    # Paths
    'data_dir': 'data',
    'output_dir': '/workspace/checkpoints',
}

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
    Compute rewards for completions.
    
    Args:
        prompts: List of prompt strings
        completions: List of completion strings (model outputs)
        **kwargs: May contain 'answer' from dataset
    
    Returns:
        List of reward floats
    """
    rewards = []
    
    # Get answers from kwargs if available
    answers = kwargs.get('answer', [None] * len(prompts))
    
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        
        if predicted is not None and answer is not None:
            if abs(predicted - answer) < CONFIG['tolerance']:
                rewards.append(CONFIG['reward_correct'])
            else:
                rewards.append(CONFIG['reward_incorrect'])
        else:
            rewards.append(CONFIG['reward_incorrect'])
    
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
    """Keep only: first, second-to-last, and last checkpoint."""
    checkpoints = get_checkpoints(output_dir)
    
    if len(checkpoints) <= 3:
        return
    
    keep_indices = {0, len(checkpoints) - 2, len(checkpoints) - 1}
    
    for i, (step, path) in enumerate(checkpoints):
        if i not in keep_indices:
            print(f"    Removing old checkpoint: {path.name}")
            shutil.rmtree(path)


def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Find latest checkpoint for resumption."""
    checkpoints = get_checkpoints(output_dir)
    if checkpoints:
        return str(checkpoints[-1][1])
    return None


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
        'max_prompt_length': 256,  # Arithmetic prompts are short
        'max_steps': CONFIG['max_steps'],
        'logging_steps': CONFIG['logging_steps'],
        'save_steps': CONFIG['save_steps'],
        'save_only_model': True,
        'beta': CONFIG['beta'],
        'bf16': True,
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
    
    # Initialize trainer
    print("Initializing GRPO trainer...")
    
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Clean up old checkpoints
    print("\nCleaning up checkpoints...")
    cleanup_checkpoints(output_dir)
    
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