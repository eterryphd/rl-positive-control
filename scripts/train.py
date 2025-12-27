#!/usr/bin/env python3
"""
RL training for arithmetic task using GRPO.

This is a positive control to validate the RL pipeline before
applying to more complex tasks like interleaving.

Usage:
    python train.py --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
import os
import re
import shutil
import torch
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

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
    
    # Training
    'learning_rate': 1e-6,
    'num_train_epochs': 1,
    'per_device_train_batch_size': 2,
    'gradient_accumulation_steps': 4,
    'num_generations': 4,
    'max_steps': 100,
    'logging_steps': 10,
    'save_steps': 25,  # Checkpoint every 25 steps
    
    # GRPO specific
    'beta': 0.1,  # KL penalty coefficient
    
    # Paths - use absolute path on persistent volume
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
    
    # Get answers from kwargs if available, otherwise try to parse from prompts
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
        return  # Nothing to clean up
    
    # Indices to keep: 0 (first), -2 (second-to-last), -1 (last)
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
    
    # GRPO config
    print("\nConfiguring GRPO...")
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=CONFIG['learning_rate'],
        num_train_epochs=CONFIG['num_train_epochs'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_generations=CONFIG['num_generations'],
        max_completion_length=CONFIG['max_new_tokens'],
        max_steps=CONFIG['max_steps'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
        save_only_model=True,  # Skip optimizer state, saves disk/time
        beta=CONFIG['beta'],
        bf16=True,
        remove_unused_columns=False,  # Keep 'answer' column for reward fn
    )
    
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
    
    # Save initial model if starting fresh (not resuming)
    initial_path = output_dir / "initial"
    if not resume_from and not initial_path.exists():
        print("\nSaving initial model (before training)...")
        trainer.save_model(str(initial_path))
        print(f"    Saved to: {initial_path}")
    
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Clean up old checkpoints (keep first, second-to-last, last)
    print("\nCleaning up checkpoints...")
    cleanup_checkpoints(output_dir)
    
    # Save final model separately for easy evaluation
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    print(f"Final model saved to: {final_path}")
    
    # List remaining checkpoints
    remaining = get_checkpoints(output_dir)
    print(f"\nCheckpoints retained: {[c[1].name for c in remaining]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()