#!/usr/bin/env python3
"""
Evaluate model on test set.

Task-agnostic evaluation script. All task-specific logic (answer extraction,
correctness checking) comes from the generator class.

Saves full prompt and response for each problem for reproducibility.

Usage:
    # Evaluate base model
    python evaluate.py --model meta-llama/Llama-3.1-8B-Instruct
    
    # Evaluate trained checkpoint
    python evaluate.py --checkpoint /workspace/checkpoints/checkpoint-100

Manual validation (paste into HuggingFace chat):
    System message: You are a calculator. Output only the number.
    User message: 35 + 17 - 8
    Expected output: 44
"""

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import utilities
from utils import build_prompt

# Import generator infrastructure
from generators.base import ProblemGenerator
from problem_generator import ArithmeticGenerator

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Token limits - match train.py for consistency
    'max_new_tokens': 4096,
    
    # Paths
    'data_dir': 'data',
    'results_dir': 'results',
    
    # Task selection - must match train.py
    'generator_class': ArithmeticGenerator,
}


def evaluate_single(model, tokenizer, problem: dict, generator: ProblemGenerator, device: str) -> dict:
    """Evaluate model on single problem using generator's task-specific logic."""
    
    prompt = build_prompt(problem['problem'], tokenizer, generator.system_message)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_length = input_ids.shape[1]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG['max_new_tokens'],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only generated tokens
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Use generator for task-specific extraction and evaluation
    predicted = generator.extract_answer(response)
    correct_answer = problem['answer']
    is_correct = generator.check_correct(predicted, correct_answer)
    reward = generator.compute_reward(predicted, correct_answer)
    
    return {
        'problem': problem['problem'],
        'prompt': prompt,
        'correct_answer': correct_answer,
        'predicted_answer': predicted,
        'model_response': response,
        'is_correct': is_correct,
        'reward': reward,
    }


def load_model(model_name: str = None, checkpoint_path: str = None, device: str = 'cuda'):
    """Load model from HuggingFace or checkpoint directory."""
    
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        print(f"Loading checkpoint: {checkpoint_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        # Try checkpoint tokenizer, fall back to base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except:
            config_path = checkpoint_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                base_model = config.get("_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")
                print(f"  Loading tokenizer from base: {base_model}")
                tokenizer = AutoTokenizer.from_pretrained(base_model)
            else:
                raise ValueError(f"Cannot determine tokenizer for {checkpoint_path}")
    else:
        print(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def evaluate_dataset(model, tokenizer, dataset: list, generator: ProblemGenerator, device: str) -> dict:
    """Evaluate model on entire dataset."""
    results = []
    correct = 0
    total_reward = 0.0
    
    for problem in tqdm(dataset, desc="Evaluating"):
        try:
            result = evaluate_single(model, tokenizer, problem, generator, device)
            results.append(result)
            if result['is_correct']:
                correct += 1
            total_reward += result['reward']
        except Exception as e:
            print(f"Error on '{problem['problem']}': {e}")
            results.append({
                'problem': problem['problem'],
                'correct_answer': problem['answer'],
                'predicted_answer': None,
                'model_response': f"ERROR: {e}",
                'is_correct': False,
                'reward': -1.0,
            })
            total_reward += -1.0
    
    accuracy = correct / len(dataset) if dataset else 0
    avg_reward = total_reward / len(dataset) if dataset else 0
    
    return {
        'accuracy': accuracy,
        'avg_reward': avg_reward,
        'correct': correct,
        'total': len(dataset),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='HuggingFace model name')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if not args.model and not args.checkpoint:
        parser.error("Must specify either --model or --checkpoint")
    
    # Instantiate generator
    generator_class = CONFIG['generator_class']
    generator: ProblemGenerator = generator_class()
    
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print(f"Task: {generator.task_name}")
    print(f"System message: {generator.system_message}")
    
    # Load dataset
    dataset_path = Path(CONFIG['data_dir']) / f'{args.dataset}.json'
    print(f"\nLoading dataset: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples")
    
    # Load model
    model, tokenizer = load_model(args.model, args.checkpoint, args.device)
    model_identifier = args.checkpoint if args.checkpoint else args.model
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_dataset(model, tokenizer, dataset, generator, args.device)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy:   {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"Avg Reward: {results['avg_reward']:.3f}")
    
    # Build output with metadata
    example_prompt = build_prompt("123 * 45", tokenizer, generator.system_message)
    
    output = {
        'metadata': {
            'model': str(model_identifier),
            'task': generator.task_name,
            'dataset': args.dataset,
            'timestamp': datetime.now().isoformat(),
            'system_message': generator.system_message,
            'max_new_tokens': CONFIG['max_new_tokens'],
            'example_prompt': example_prompt,
        },
        'summary': {
            'accuracy': results['accuracy'],
            'avg_reward': results['avg_reward'],
            'correct': results['correct'],
            'total': results['total'],
        },
        'results': results['results']
    }
    
    # Save results
    output_dir = Path(CONFIG['results_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.checkpoint:
        model_name = Path(args.checkpoint).name
    else:
        model_name = args.model.split('/')[-1]
    
    output_file = output_dir / f"eval_{model_name}_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Show sample results
    print(f"\n{'=' * 70}")
    print("SAMPLE RESULTS")
    print("=" * 70)
    for result in results['results'][:5]:
        print(generator.format_example(
            {'problem': result['problem'], 'answer': result['correct_answer']},
            result['model_response'],
            result['predicted_answer']
        ))
        print()


if __name__ == "__main__":
    main()