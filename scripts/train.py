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
import re
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
    'per_device_train_batch_size': 4,
    'gradient_accumulation_steps': 4,
    'num_generations': 4,  # G in GRPO - responses per prompt
    'max_steps': 500,
    'logging_steps': 10,
    'save_steps': 100,
    
    # GRPO specific
    'beta': 0.1,  # KL penalty coefficient
    
    # Paths
    'data_dir': 'data',
    'output_dir': 'checkpoints',
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
    
    # GRPO config
    print("\nConfiguring GRPO...")
    training_args = GRPOConfig(
        output_dir=CONFIG['output_dir'],
        learning_rate=CONFIG['learning_rate'],
        num_train_epochs=CONFIG['num_train_epochs'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_generations=CONFIG['num_generations'],
        max_completion_length=CONFIG['max_new_tokens'],
        max_steps=CONFIG['max_steps'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
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
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    final_path = Path(CONFIG['output_dir']) / "final"
    trainer.save_model(str(final_path))
    print(f"Model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()