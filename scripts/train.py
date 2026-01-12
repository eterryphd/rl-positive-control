#!/usr/bin/env python3
"""
RL training - task-agnostic GRPO pipeline.

The training script is completely task-agnostic. All task-specific logic
(problem generation, answer extraction, reward computation) is handled
by the generator class specified in CONFIG.

Currently configured for arithmetic positive control.
To switch to interleaving: change CONFIG['generator_class'].

TESTED STACK (HuggingFace Cookbook, Dec 2025):
    trl==0.23.1, vllm==0.11.0, transformers==4.57.0

OPTIMIZATIONS:
    - Waits for vLLM AFTER DeepSpeed init (parallel model loading)
    - Saves 6-8 minutes of startup time

Usage (with vLLM server mode - recommended for 4x A40):
    # Terminal 1: Start vLLM server on GPU 3
    CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model meta-llama/Llama-3.1-8B-Instruct
    
    # Terminal 2: Launch training on GPUs 0,1,2
    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \\
        --config_file accelerate_config.yaml \\
        --num_processes 3 \\
        train.py --model meta-llama/Llama-3.1-8B-Instruct

Usage (without vLLM - slower but simpler):
    accelerate launch --config_file accelerate_config.yaml train.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --no-vllm
"""

import argparse
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Set, Optional

# Set cache directories before importing HF libraries
# This prevents filling up the root filesystem on Runpod
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
if 'PIP_CACHE_DIR' not in os.environ:
    os.environ['PIP_CACHE_DIR'] = '/workspace/.cache/pip'

from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import torch

# Import utilities (task-agnostic only)
from utils import build_prompt

# Import generator infrastructure
from generators.base import ProblemGenerator
from problem_generator import ArithmeticGenerator

# ============================================================================
# CONFIG - User configurable settings
# ============================================================================

CONFIG = {
    # Token limits - sized for interleave task to validate full pipeline
    'max_new_tokens': 4096,
    'max_prompt_length': 2048,
    
    # Generation
    'temperature': 0.9,
    'num_generations': 6,
    
    # Training - tuned for 4x A40 (48GB each)
    'learning_rate': 1e-6,
    'num_train_epochs': 1,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'max_steps': 200,
    'logging_steps': 1,
    'save_steps': 20,
    
    # GRPO specific
    'beta': 0.0,  # KL penalty - 0.0 is standard (no ref model needed)
    
    # Paths
    'data_dir': 'data',
    'output_dir': '/workspace/checkpoints',
    
    # Dynamic data generation
    'dataset_size': 2500,
    'problem_config': {},  # Passed to generator
    
    # === TASK SELECTION ===
    # Change this to swap tasks (e.g., InterleaveGenerator)
    'generator_class': ArithmeticGenerator,
    
    # Validation
    'validation_size': 500,
    'validation_steps': 20,
}

# ============================================================================
# VLLM SERVER WAIT
# ============================================================================

def wait_for_vllm(host: str = 'localhost', port: int = 8000, timeout: int = 1800, check_interval: int = 5):
    """
    Wait for vLLM server to be ready.
    
    Called AFTER DeepSpeed init to allow parallel model loading.
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
            if elapsed - last_print >= 30:
                print(f"    Still waiting for vLLM... ({elapsed:.0f}s)")
                last_print = elapsed
            time.sleep(check_interval)
    
    raise RuntimeError(f"vLLM server not ready after {timeout}s")


# ============================================================================
# REWARD FUNCTION FACTORY
# ============================================================================

def make_reward_fn(generator: ProblemGenerator):
    """
    Create reward function that uses the generator's task-specific logic.
    
    This is a factory because GRPOTrainer expects a function, not a method.
    The returned function closes over the generator instance.
    """
    def reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
        """
        Compute rewards for completions.
        
        Delegates to generator.compute_reward() for task-specific logic.
        """
        rewards = []
        answers = kwargs.get('answer', [None] * len(prompts))
        
        for completion, answer in zip(completions, answers):
            predicted = generator.extract_answer(completion)
            
            if predicted is None:
                # Log extraction failures for debugging
                print(f"    Warning: Could not extract answer from: '{completion[:100]}'")
            
            reward = generator.compute_reward(predicted, answer)
            rewards.append(reward)
        
        return rewards
    
    return reward_fn


# ============================================================================
# DATA LOADING
# ============================================================================

def prepare_dataset(tokenizer, args, config: dict, generator: ProblemGenerator) -> Dataset:
    """Prepare dataset - static or dynamic based on flag."""
    
    system_message = generator.system_message
    
    if args.dynamic_data:
        print(f">>> Dynamic data mode - generating {config['dataset_size']} problems")
        print(f"    Task: {generator.task_name}")
        
        problems = generator.generate_batch(
            config['dataset_size'],
            config.get('problem_config', {})
        )
        
        data = {
            'prompt': [build_prompt(p['problem'], tokenizer, system_message) for p in problems],
            'problem': [p['problem'] for p in problems],
            'answer': [p['answer'] for p in problems],
        }
        
        print(f"    Generated {len(problems)} unique problems")
        return Dataset.from_dict(data)
    
    else:
        print(">>> Static data mode - loading from file")
        data_path = Path(config['data_dir']) / 'train.json'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        with open(data_path) as f:
            problems = json.load(f)
        
        data = {
            'prompt': [build_prompt(p['problem'], tokenizer, system_message) for p in problems],
            'problem': [p['problem'] for p in problems],
            'answer': [p['answer'] for p in problems],
        }
        
        print(f"    Loaded {len(problems)} problems from {data_path}")
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
    """Keep only: first, second-to-last, and last valid checkpoint."""
    checkpoints = get_checkpoints(output_dir)
    
    # Remove invalid checkpoints first
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
    required_files = ['trainer_state.json']
    
    for f in required_files:
        if not (checkpoint_path / f).exists():
            return False
    
    # Check for model files
    has_model_files = any(
        checkpoint_path.glob('*.safetensors')
    ) or any(
        checkpoint_path.glob('*.bin')
    ) or (checkpoint_path / 'pytorch_model.bin').exists()
    
    # For DeepSpeed, check for ZeRO checkpoint
    has_deepspeed = (checkpoint_path / 'zero_to_fp32.py').exists() or \
                    any(checkpoint_path.glob('global_step*'))
    
    return has_model_files or has_deepspeed


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find latest valid checkpoint for resumption."""
    checkpoints = get_checkpoints(output_dir)
    
    for step, path in reversed(checkpoints):
        if validate_checkpoint(path):
            return str(path)
        else:
            print(f"    Warning: Checkpoint {path.name} is incomplete, skipping...")
    
    return None


# ============================================================================
# CALLBACKS
# ============================================================================

class CheckpointCleanupCallback(TrainerCallback):
    """Clean up old checkpoints after each save."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            cleanup_checkpoints(self.output_dir)


class RewardThresholdSaveCallback(TrainerCallback):
    """Save checkpoints when reward exceeds threshold or is new best."""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.best_reward = float('-inf')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        reward = logs.get('reward', None)
        if reward is None:
            return
        
        if reward > self.threshold or reward > self.best_reward:
            if reward > self.best_reward:
                self.best_reward = reward
                print(f">>> New best reward: {reward:.3f} - saving checkpoint")
            else:
                print(f">>> Reward {reward:.3f} > {self.threshold} threshold - saving checkpoint")
            control.should_save = True


class ValidationCallback(TrainerCallback):
    """
    Periodic validation on held-out set.
    
    Uses generator methods for task-specific evaluation.
    """
    
    def __init__(self, validation_size: int, validation_steps: int,
                 generator: ProblemGenerator, tokenizer, config: dict):
        self.validation_size = validation_size
        self.validation_steps = validation_steps
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config
        self.seen_problems: Set[str] = set()
        self.initialized = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Capture training problems to exclude from validation."""
        trainer = kwargs.get('trainer', None)
        if trainer and hasattr(trainer, 'train_dataset'):
            dataset = trainer.train_dataset
            if 'problem' in dataset.column_names:
                for problem in dataset['problem']:
                    self.seen_problems.add(problem)
                print(f">>> Validation: Tracking {len(self.seen_problems)} seen problems")
                self.initialized = True
    
    def on_step_end(self, args, state, control, **kwargs):
        """Validate every N steps on held-out problems."""
        if state.global_step % self.validation_steps != 0:
            return
        if not state.is_world_process_zero:
            return
        
        print(f"\n>>> Validation at step {state.global_step}...")
        
        # Generate held-out problems
        val_problems = self.generator.generate_held_out(
            self.validation_size,
            self.seen_problems,
            self.config.get('problem_config', {})
        )
        
        if not val_problems:
            print("    Warning: Could not generate validation problems")
            return
        
        # Get model and device
        trainer = kwargs.get('trainer', None)
        if trainer is None or not hasattr(trainer, 'model'):
            print("    Warning: Cannot access model for validation")
            return
        
        model = trainer.model
        device = model.device if hasattr(model, 'device') else 'cuda'
        system_message = self.generator.system_message
        
        # Evaluate
        correct = 0
        for problem in val_problems:
            try:
                prompt = build_prompt(problem['problem'], self.tokenizer, system_message)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config['max_new_tokens'],
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                predicted = self.generator.extract_answer(response)
                if self.generator.check_correct(predicted, problem['answer']):
                    correct += 1
                    
            except Exception as e:
                print(f"    Error evaluating: {e}")
                continue
        
        accuracy = correct / len(val_problems) if val_problems else 0
        print(f"    Accuracy: {accuracy:.1%} ({correct}/{len(val_problems)})")


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    print("=" * 70)
    print("GRPO TRAINING - TASK-AGNOSTIC PIPELINE")
    print("=" * 70)
    
    # Instantiate generator
    generator_class = CONFIG['generator_class']
    generator: ProblemGenerator = generator_class()
    print(f"\nTask: {generator.task_name}")
    print(f"System message: {generator.system_message}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(tokenizer, args, CONFIG, generator)
    print(f"    Train: {len(train_dataset)} examples")
    
    # Output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for checkpoint to resume
    resume_from = find_latest_checkpoint(output_dir)
    if resume_from:
        print(f"\n>>> Found checkpoint: {resume_from}")
        print("    Will resume training.")
    
    # GRPO config
    print("\nConfiguring GRPO...")
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=CONFIG['learning_rate'],
        num_train_epochs=CONFIG['num_train_epochs'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_generations=CONFIG['num_generations'],
        max_completion_length=CONFIG['max_new_tokens'],
        max_prompt_length=CONFIG['max_prompt_length'],
        temperature=CONFIG['temperature'],
        max_steps=CONFIG['max_steps'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
        lr_scheduler_type='constant',
        save_only_model=False,  # Required for DeepSpeed resume
    )
    
    # Create reward function using generator
    reward_fn = make_reward_fn(generator)
    
    # GRPO trainer
    print("\nInitializing GRPO trainer...")
    grpo_trainer = GRPOTrainer(
        model=args.model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_fn=reward_fn,
        callbacks=[
            RewardThresholdSaveCallback(),
            CheckpointCleanupCallback(output_dir),
            ValidationCallback(
                CONFIG['validation_size'],
                CONFIG['validation_steps'],
                generator,
                tokenizer,
                CONFIG
            )
        ],
        resume_from_checkpoint=resume_from,
    )
    
    # Wait for vLLM if using it
    if not args.no_vllm:
        print("\n>>> Trainer initialized, waiting for vLLM server...")
        wait_for_vllm(host='localhost', port=args.vllm_port)
    
    # Train
    print("\nStarting training...")
    grpo_trainer.train()
    
    print("\nTraining complete!")
    print(f"Checkpoints in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--no-vllm', action='store_true', help='Disable vLLM server mode')
    parser.add_argument('--vllm-port', type=int, default=8000, help='vLLM server port')
    parser.add_argument('--dynamic-data', action='store_true', help='Generate data dynamically')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()