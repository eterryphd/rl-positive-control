#!/usr/bin/env python3
"""
Generate arithmetic problem datasets for RL training.

Creates:
- 2000 training examples (4-step arithmetic with +, -, *, /)
- 250 validation examples
- 250 test examples
"""

import json
import random
from pathlib import Path

def generate_problem(n_ops=4, operations=['+', '-', '*', '/']):
    """Generate single arithmetic problem with n_ops operations."""
    # Generate numbers (5-50 range, avoid 0 for division)
    nums = [random.randint(5, 50) for _ in range(n_ops + 1)]
    ops = [random.choice(operations) for _ in range(n_ops)]
    
    # Build problem string
    problem_str = str(nums[0])
    for op, num in zip(ops, nums[1:]):
        problem_str += f" {op} {num}"
    
    # Calculate answer with proper order of operations
    try:
        answer = eval(problem_str)
        # Round to 2 decimals to avoid floating point issues
        if isinstance(answer, float):
            answer = round(answer, 2)
    except ZeroDivisionError:
        # Regenerate if division by zero
        return generate_problem(n_ops, operations)
    
    return {
        'problem': problem_str,
        'answer': answer,
        'n_ops': n_ops
    }

def generate_dataset(n_examples, n_ops=4, operations=['+', '-', '*', '/']):
    """Generate dataset of n_examples problems."""
    return [generate_problem(n_ops, operations) for _ in range(n_examples)]

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("Generating datasets...")
    
    # Generate datasets
    train_data = generate_dataset(2000, n_ops=4)
    val_data = generate_dataset(250, n_ops=4)
    test_data = generate_dataset(250, n_ops=4)
    
    # Save to JSON
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for name, data in datasets.items():
        filepath = data_dir / f'{name}.json'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(data)} examples to {filepath}")
    
    # Print sample problems
    print("\n=== Sample Problems ===")
    for i, problem in enumerate(train_data[:5]):
        print(f"{i+1}. {problem['problem']} = {problem['answer']}")
    
    print("\n✓ Dataset generation complete!")
    print(f"  Train: 2000 examples")
    print(f"  Val:   250 examples")
    print(f"  Test:  250 examples")

if __name__ == "__main__":
    main()
