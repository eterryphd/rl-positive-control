#!/usr/bin/env python3
"""
RL training for arithmetic task using PPO.

This is a positive control to validate the RL pipeline before
applying to more complex tasks like interleaving.

Usage:
    python train.py --model meta-llama/Llama-3.2-3B-Instruct

Checkpoint format (compatible with evaluate.py):
    {
        'model_state_dict': model.state_dict(),
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'step': step,
        'config': CONFIG,
    }
"""

import argparse
import json
import re
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset

# ============================================================================
# CONFIG - User configurable settings
# ============================================================================

CONFIG = {
    # Prompt configuration - MUST MATCH evaluate.py
    'system_message': "You are a calculator. Output only the number.",
    'max_new_tokens': 15,  # Shorter for training efficiency
    
    # Reward
    'reward_correct': 1.0,
    'reward_incorrect': 0.0,
    'tolerance': 0.01,
    
    # Training
    'learning_rate': 1e-5,
    'batch_size': 4,
    'mini_batch_size': 2,
    'gradient_accumulation_steps': 2,
    'ppo_epochs': 4,
    'max_steps': 1000,
    'eval_every': 100,
    'save_every': 200,
    
    # Paths
    'data_dir': 'data',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    
    # KL penalty
    'init_kl_coef': 0.2,
    'target_kl': 6.0,
}

# ============================================================================
# PROMPT AND REWARD
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


def extract_answer(response: str):
    """Extract numeric answer from model response."""
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def compute_reward(response: str, correct_answer: float) -> float:
    """Compute reward for a response."""
    predicted = extract_answer(response)
    
    if predicted is not None:
        if abs(predicted - correct_answer) < CONFIG['tolerance']:
            return CONFIG['reward_correct']
    
    return CONFIG['reward_incorrect']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_data(tokenizer) -> Dataset:
    """Load training data and format for PPO."""
    data_path = Path(CONFIG['data_dir']) / 'train.json'
    
    with open(data_path) as f:
        problems = json.load(f)
    
    # Build prompts
    data = {
        'query': [build_prompt(p['problem'], tokenizer) for p in problems],
        'problem': [p['problem'] for p in problems],
        'answer': [p['answer'] for p in problems],
    }
    
    return Dataset.from_dict(data)


def load_eval_data(tokenizer) -> list:
    """Load validation data."""
    data_path = Path(CONFIG['data_dir']) / 'val.json'
    
    with open(data_path) as f:
        problems = json.load(f)
    
    return [
        {
            'query': build_prompt(p['problem'], tokenizer),
            'problem': p['problem'],
            'answer': p['answer'],
        }
        for p in problems
    ]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, tokenizer, eval_data: list, device: str) -> dict:
    """Run evaluation on validation set."""
    model.eval()
    correct = 0
    results = []
    
    for item in tqdm(eval_data[:50], desc="Evaluating", leave=False):  # Subset for speed
        inputs = tokenizer(item['query'], return_tensors="pt").to(device)
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG['max_new_tokens'],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        reward = compute_reward(response, item['answer'])
        is_correct = reward == CONFIG['reward_correct']
        
        if is_correct:
            correct += 1
        
        results.append({
            'problem': item['problem'],
            'answer': item['answer'],
            'response': response,
            'correct': is_correct,
        })
    
    model.train()
    
    accuracy = correct / len(results)
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(results),
        'results': results,
    }


# ============================================================================
# TRAINING
# ============================================================================

def save_checkpoint(model, tokenizer, base_model: str, step: int, eval_result: dict):
    """Save checkpoint compatible with evaluate.py."""
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"step_{step}.pt"
    
    # Get the base model (unwrap value head)
    if hasattr(model, 'pretrained_model'):
        base = model.pretrained_model
    else:
        base = model
    
    torch.save({
        'model_state_dict': base.state_dict(),
        'base_model': base_model,
        'step': step,
        'config': CONFIG,
        'eval_accuracy': eval_result['accuracy'] if eval_result else None,
        'timestamp': datetime.now().isoformat(),
    }, checkpoint_path)
    
    print(f"    Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def train(args):
    """Main training loop."""
    print("=" * 70)
    print("RL TRAINING - ARITHMETIC POSITIVE CONTROL")
    print("=" * 70)
    
    device = args.device
    
    # ========== SETUP ==========
    print(f"\nModel: {args.model}")
    print(f"Device: {device}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head for PPO
    print("Loading model with value head...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    # Load reference model (frozen, for KL penalty)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    # Load data
    print("\nLoading data...")
    train_dataset = load_train_data(tokenizer)
    eval_data = load_eval_data(tokenizer)
    print(f"    Train: {len(train_dataset)} examples")
    print(f"    Eval: {len(eval_data)} examples")
    
    # PPO config
    ppo_config = PPOConfig(
        learning_rate=CONFIG['learning_rate'],
        batch_size=CONFIG['batch_size'],
        mini_batch_size=CONFIG['mini_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        ppo_epochs=CONFIG['ppo_epochs'],
        init_kl_coef=CONFIG['init_kl_coef'],
        target_kl=CONFIG['target_kl'],
        log_with=None,  # Could use 'wandb'
    )
    
    # Initialize trainer
    print("\nInitializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
    )
    
    # ========== INITIAL EVAL ==========
    print("\n" + "-" * 70)
    print("Initial evaluation...")
    eval_result = evaluate(model.pretrained_model, tokenizer, eval_data, device)
    print(f"    Baseline accuracy: {eval_result['accuracy']:.1%}")
    
    # Setup logging
    log_dir = Path(CONFIG['log_dir'])
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    
    # ========== TRAINING LOOP ==========
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    step = 0
    best_accuracy = eval_result['accuracy']
    
    for epoch in range(CONFIG['max_steps'] // len(train_dataset) + 1):
        for batch in ppo_trainer.dataloader:
            if step >= CONFIG['max_steps']:
                break
            
            # Get queries and answers
            queries = batch['query']
            answers = batch['answer']
            
            # Tokenize queries
            query_tensors = [
                tokenizer(q, return_tensors="pt")['input_ids'].squeeze().to(device)
                for q in queries
            ]
            
            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=CONFIG['max_new_tokens'],
                do_sample=True,  # Need sampling for exploration
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decode responses (just the generated part)
            responses = []
            for query_tensor, response_tensor in zip(query_tensors, response_tensors):
                generated = response_tensor[len(query_tensor):]
                response = tokenizer.decode(generated, skip_special_tokens=True)
                responses.append(response)
            
            # Compute rewards
            rewards = [
                torch.tensor(compute_reward(resp, ans), dtype=torch.float32).to(device)
                for resp, ans in zip(responses, answers)
            ]
            
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            log_entry = {
                'step': step,
                'mean_reward': mean_reward,
                'kl': stats.get('objective/kl', 0),
                'entropy': stats.get('objective/entropy', 0),
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Progress
            if step % 10 == 0:
                print(f"Step {step}: reward={mean_reward:.2f}, kl={stats.get('objective/kl', 0):.4f}")
            
            # Eval
            if step > 0 and step % CONFIG['eval_every'] == 0:
                print(f"\n--- Evaluation at step {step} ---")
                eval_result = evaluate(model.pretrained_model, tokenizer, eval_data, device)
                print(f"    Accuracy: {eval_result['accuracy']:.1%} (baseline: {best_accuracy:.1%})")
                
                if eval_result['accuracy'] > best_accuracy:
                    best_accuracy = eval_result['accuracy']
                    print(f"    New best! Saving...")
                    save_checkpoint(model, tokenizer, args.model, step, eval_result)
            
            # Save periodic checkpoint
            if step > 0 and step % CONFIG['save_every'] == 0:
                save_checkpoint(model, tokenizer, args.model, step, eval_result)
            
            step += 1
    
    # ========== FINAL ==========
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    # Final eval
    print("\nFinal evaluation...")
    eval_result = evaluate(model.pretrained_model, tokenizer, eval_data, device)
    print(f"    Final accuracy: {eval_result['accuracy']:.1%}")
    print(f"    Best accuracy: {best_accuracy:.1%}")
    
    # Save final checkpoint
    save_checkpoint(model, tokenizer, args.model, step, eval_result)
    
    print(f"\nLogs: {log_file}")
    print(f"Checkpoints: {CONFIG['checkpoint_dir']}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()