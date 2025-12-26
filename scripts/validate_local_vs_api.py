#!/usr/bin/env python3
"""
Validate that local model inference matches API inference.

Run once before trusting local evaluation for RL training.
Compares same problems through both paths, reports discrepancies.

Saves full prompt and response for each problem for reproducibility.

Usage:
    python validate_local_vs_api.py --model meta-llama/Llama-3.2-3B-Instruct --n 20

Manual validation (paste into HuggingFace chat):
    System message: You are a calculator. Output only the number.
    User message: 35 + 17 - 8
    Expected output: 44
"""

import argparse
import json
import os
import re
import torch
from pathlib import Path
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

# ============================================================================
# CONFIG - User configurable settings
# Must match evaluate.py for valid comparison
# ============================================================================

CONFIG = {
    # Prompt configuration - MUST MATCH evaluate.py
    'system_message': "You are a calculator. Output only the number.",
    'max_new_tokens': 150,
    
    # Paths
    'default_dataset': 'data/test.json',
    'results_dir': 'results',
    'output_filename': 'validation_local_vs_api.json',
    
    # API settings
    'default_provider': 'novita',
    
    # Comparison settings
    'tolerance': 0.01,  # For floating point comparison
    'default_n': 20,    # Number of problems to test
}

# Expose for module use
SYSTEM_MESSAGE = CONFIG['system_message']
MAX_NEW_TOKENS = CONFIG['max_new_tokens']


def extract_answer(response: str) -> Optional[float]:
    """Extract numeric answer from model response."""
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def check_correct(predicted: Optional[float], correct: float) -> bool:
    """Check if prediction matches within tolerance."""
    if predicted is None:
        return False
    return abs(predicted - correct) < CONFIG['tolerance']


# === LOCAL INFERENCE ===

def load_local_model(model_name: str, device: str = 'cuda'):
    """Load model for local inference."""
    print(f"Loading local model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def run_local(model, tokenizer, problem: str, device: str = 'cuda') -> tuple[str, str]:
    """Run inference locally. Returns (response, prompt)."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": problem}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, prompt


# === API INFERENCE ===

def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or Colab."""
    try:
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    except:
        return os.getenv("HF_TOKEN")


def run_api(client: InferenceClient, model: str, problem: str) -> str:
    """Run inference via API."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": problem}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )
    return response.choices[0].message.content


# === COMPARISON ===

def compare_single(problem: Dict, local_response: str, api_response: str, prompt: str) -> Dict:
    """Compare local vs API for single problem."""
    correct_answer = problem['answer']
    
    local_parsed = extract_answer(local_response)
    api_parsed = extract_answer(api_response)
    
    local_correct = check_correct(local_parsed, correct_answer)
    api_correct = check_correct(api_parsed, correct_answer)
    
    # Check if they match each other (regardless of correctness)
    if local_parsed is not None and api_parsed is not None:
        outputs_match = abs(local_parsed - api_parsed) < CONFIG['tolerance']
    else:
        outputs_match = (local_parsed is None and api_parsed is None)
    
    return {
        'problem': problem['problem'],
        'prompt': prompt,
        'correct_answer': correct_answer,
        'local_response': local_response,
        'local_parsed': local_parsed,
        'local_correct': local_correct,
        'api_response': api_response,
        'api_parsed': api_parsed,
        'api_correct': api_correct,
        'outputs_match': outputs_match,
    }


def print_result(result: Dict, verbose: bool = False):
    """Print single comparison result."""
    match_symbol = "✓" if result['outputs_match'] else "✗"
    local_symbol = "✓" if result['local_correct'] else "✗"
    api_symbol = "✓" if result['api_correct'] else "✗"
    
    print(f"{match_symbol} {result['problem']} = {result['correct_answer']}")
    print(f"   Local [{local_symbol}]: '{result['local_response']}' → {result['local_parsed']}")
    print(f"   API   [{api_symbol}]: '{result['api_response']}' → {result['api_parsed']}")
    
    if not result['outputs_match']:
        print(f"   ⚠️  MISMATCH")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name')
    parser.add_argument('--dataset', type=str, default=CONFIG['default_dataset'],
                        help='Path to test dataset')
    parser.add_argument('--n', type=int, default=CONFIG['default_n'],
                        help='Number of problems to test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--provider', type=str, default=CONFIG['default_provider'],
                        help='API provider')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each comparison')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LOCAL VS API VALIDATION")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"System message: {SYSTEM_MESSAGE}")
    print(f"Max tokens: {MAX_NEW_TOKENS}")
    print(f"Provider: {args.provider}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}")
    with open(args.dataset) as f:
        dataset = json.load(f)
    
    # Take subset
    problems = dataset[:args.n]
    print(f"Testing {len(problems)} problems")
    
    # Setup local
    print("\n" + "-" * 70)
    print("Setting up local model...")
    model, tokenizer = load_local_model(args.model, args.device)
    
    # Setup API
    print("\nSetting up API client...")
    hf_token = get_hf_token()
    if not hf_token:
        print("✗ HF_TOKEN not found")
        return
    client = InferenceClient(api_key=hf_token, provider=args.provider)
    print("✓ Ready")
    
    # Run comparison
    print("\n" + "-" * 70)
    print("Running comparison...")
    print("-" * 70 + "\n")
    
    results = []
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem['problem']}", end='', flush=True)
        
        local_response, prompt = run_local(model, tokenizer, problem['problem'], args.device)
        api_response = run_api(client, args.model, problem['problem'])
        
        result = compare_single(problem, local_response, api_response, prompt)
        results.append(result)
        
        if result['outputs_match']:
            print(" ✓")
        else:
            print(" ✗ MISMATCH")
        
        if args.verbose:
            print_result(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_match = sum(1 for r in results if r['outputs_match'])
    n_local_correct = sum(1 for r in results if r['local_correct'])
    n_api_correct = sum(1 for r in results if r['api_correct'])
    
    print(f"\nOutputs match:    {n_match}/{len(results)} ({100*n_match/len(results):.1f}%)")
    print(f"Local correct:    {n_local_correct}/{len(results)} ({100*n_local_correct/len(results):.1f}%)")
    print(f"API correct:      {n_api_correct}/{len(results)} ({100*n_api_correct/len(results):.1f}%)")
    
    if n_match == len(results):
        print("\n✅ LOCAL AND API MATCH PERFECTLY")
        print("   Safe to use local evaluation for RL training.")
    else:
        print(f"\n⚠️  {len(results) - n_match} MISMATCHES FOUND")
        print("   Investigate before trusting local evaluation.")
        
        # Show mismatches
        print("\nMismatched problems:")
        for r in results:
            if not r['outputs_match']:
                print_result(r)
    
    # Save detailed results
    # Generate example prompt for reproducibility
    example_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": "35 + 17 - 8"}
    ]
    example_prompt = tokenizer.apply_chat_template(
        example_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    output_path = Path(CONFIG['results_dir']) / CONFIG['output_filename']
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'model': args.model,
                'system_message': SYSTEM_MESSAGE,
                'max_new_tokens': MAX_NEW_TOKENS,
                'provider': args.provider,
                'n_problems': len(problems),
                'example_prompt': example_prompt,
            },
            'summary': {
                'outputs_match': n_match,
                'local_correct': n_local_correct,
                'api_correct': n_api_correct,
                'total': len(results),
            },
            'results': results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()