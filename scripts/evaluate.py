#!/usr/bin/env python3
"""
Evaluate model on arithmetic test set.

Usage:
    # Evaluate base model
    python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct
    
    # Evaluate trained checkpoint
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def extract_answer(response):
    """Extract numeric answer from model response."""
    # Try to find numbers (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', response)
    
    if numbers:
        try:
            return float(numbers[-1])  # Take last number
        except ValueError:
            return None
    return None

def evaluate_single(model, tokenizer, problem, device='cuda'):
    """Evaluate model on single problem."""
    prompt = f"Solve this arithmetic problem: {problem['problem']}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response[len(prompt):].strip()
    
    predicted = extract_answer(response)
    correct_answer = problem['answer']
    
    # Check if close enough (within 0.01 for floating point)
    if predicted is not None and correct_answer is not None:
        is_correct = abs(predicted - correct_answer) < 0.01
    else:
        is_correct = False
    
    return {
        'problem': problem['problem'],
        'correct_answer': correct_answer,
        'model_response': response,
        'predicted_answer': predicted,
        'is_correct': is_correct
    }

def load_model(model_name=None, checkpoint_path=None, device='cuda'):
    """Load model from HuggingFace or checkpoint."""
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # Load checkpoint (we'll implement saving/loading in train script)
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        tokenizer = checkpoint['tokenizer']
    else:
        print(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def evaluate_dataset(model, tokenizer, dataset, device='cuda'):
    """Evaluate model on entire dataset."""
    results = []
    correct = 0
    
    for problem in tqdm(dataset, desc="Evaluating"):
        result = evaluate_single(model, tokenizer, problem, device)
        results.append(result)
        if result['is_correct']:
            correct += 1
    
    accuracy = correct / len(dataset)
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(dataset),
        'results': results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='HuggingFace model name')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='test', 
                       choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if not args.model and not args.checkpoint:
        parser.error("Must specify either --model or --checkpoint")
    
    # Load dataset
    dataset_path = Path(f'data/{args.dataset}.json')
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"✓ Loaded {len(dataset)} examples")
    
    # Load model
    model, tokenizer = load_model(args.model, args.checkpoint, args.device)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    results = evaluate_dataset(model, tokenizer, dataset, args.device)
    
    # Print results
    print(f"\nAccuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    
    # Save detailed results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    model_name = args.checkpoint if args.checkpoint else args.model.split('/')[-1]
    output_file = output_dir / f"eval_{model_name}_{args.dataset}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {output_file}")
    
    # Show some examples
    print("\n=== Sample Results ===")
    for i, result in enumerate(results['results'][:5]):
        status = "✓" if result['is_correct'] else "✗"
        print(f"{status} {result['problem']} = {result['correct_answer']}")
        print(f"   Model said: {result['predicted_answer']}")
        print(f"   Response: {result['model_response'][:80]}...")
        print()

if __name__ == "__main__":
    main()
