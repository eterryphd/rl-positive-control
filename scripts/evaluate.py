#!/usr/bin/env python3
"""
Evaluate model on arithmetic test set.

Saves full prompt and response for each problem for reproducibility.

Usage:
    # Evaluate base model
    python evaluate.py --model meta-llama/Llama-3.1-8B-Instruct
    
    # Evaluate trained checkpoint (HuggingFace directory)
    python evaluate.py --checkpoint /workspace/checkpoints/checkpoint-100

Manual validation (paste into HuggingFace chat):
    System message: You are a calculator. Output only the number.
    User message: 35 + 17 - 8
    Expected output: 44

Actual prompt after chat template (for debugging):
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a calculator. Output only the number.<|eot_id|><|start_header_id|>user<|end_header_id|>

    35 + 17 - 8<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import centralized utilities (including system message and build_prompt)
from utils import extract_answer, build_prompt, SYSTEM_MESSAGE

# ============================================================================
# CONFIG - User configurable settings
# ============================================================================
CONFIG = {
    # Prompt configuration - system message centralized in utils.py
    'max_new_tokens': 150,
    
    # Paths
    'data_dir': 'data',
    'results_dir': 'results',
    
    # Answer extraction
    'tolerance': 0.01,  # For floating point comparison
}


# Expose for imports
MAX_NEW_TOKENS = CONFIG['max_new_tokens']


def evaluate_single(model, tokenizer, problem: dict, device: str) -> dict:
    """Evaluate model on single problem."""
    prompt = build_prompt(problem['problem'], tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_length = input_ids.shape[1]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated tokens
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    predicted = extract_answer(response)
    correct_answer = problem['answer']
    is_correct = (
        predicted is not None and 
        abs(predicted - correct_answer) < CONFIG['tolerance']
    )
    
    return {
        'problem': problem['problem'],
        'prompt': prompt,
        'correct_answer': correct_answer,
        'predicted_answer': predicted,
        'model_response': response,
        'is_correct': is_correct
    }


def load_model(model_name: str = None, checkpoint_path: str = None, device: str = 'cuda'):
    """Load model from HuggingFace or checkpoint directory."""
    
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # HuggingFace checkpoint directory (from Trainer/GRPO)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        # Try to load tokenizer from checkpoint, fall back to base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except:
            # Checkpoint may not have tokenizer - get from config
            config_path = checkpoint_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                base_model = config.get("_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")
                print(f"  Loading tokenizer from base model: {base_model}")
                tokenizer = AutoTokenizer.from_pretrained(base_model)
            else:
                raise ValueError(f"Cannot determine tokenizer for checkpoint {checkpoint_path}")
        
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


def evaluate_dataset(model, tokenizer, dataset: list, device: str) -> dict:
    """Evaluate model on entire dataset."""
    results = []
    correct = 0
    
    for problem in tqdm(dataset, desc="Evaluating"):
        try:
            result = evaluate_single(model, tokenizer, problem, device)
            results.append(result)
            if result['is_correct']:
                correct += 1
        except Exception as e:
            print(f"Error on problem '{problem['problem']}': {e}")
            results.append({
                'problem': problem['problem'],
                'correct_answer': problem['answer'],
                'predicted_answer': None,
                'model_response': f"ERROR: {e}",
                'is_correct': False
            })
    
    accuracy = correct / len(dataset) if dataset else 0
    
    return {
        'accuracy': accuracy,
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
    
    # Load dataset
    dataset_path = Path(CONFIG['data_dir']) / f'{args.dataset}.json'
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} examples")
    
    # Load model
    model, tokenizer = load_model(args.model, args.checkpoint, args.device)
    model_identifier = args.checkpoint if args.checkpoint else args.model
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    results = evaluate_dataset(model, tokenizer, dataset, args.device)
    
    # Print results
    print(f"\nAccuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    
    # Build output with metadata
    # Generate example prompt for reproducibility
    example_prompt = build_prompt("35 + 17 - 8", tokenizer)
    
    output = {
        'metadata': {
            'model': str(model_identifier),
            'dataset': args.dataset,
            'timestamp': datetime.now().isoformat(),
            'system_message': SYSTEM_MESSAGE,
            'max_new_tokens': MAX_NEW_TOKENS,
            'example_prompt': example_prompt,
        },
        'summary': {
            'accuracy': results['accuracy'],
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
    print("\n=== Sample Results ===")
    for result in results['results'][:5]:
        status = "Correct" if result['is_correct'] else "Incorrect"
        print(f"{status} {result['problem']} = {result['correct_answer']}")
        print(f"   Model output: '{result['model_response']}' → parsed: {result['predicted_answer']}")
        print()


if __name__ == "__main__":
    main()