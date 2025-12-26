#!/usr/bin/env python3
"""
Generate arithmetic datasets for RL training.

Produces:
    data/train.json  (2000 problems)
    data/val.json    (250 problems)
    data/test.json   (250 problems)

Each problem has format:
    {"problem": "35 + 17 - 8", "answer": 44, "n_ops": 2}
"""

import json
import random
from pathlib import Path

# ============================================================================
# CONFIG - User configurable settings
# ============================================================================

CONFIG = {
    # Dataset splits
    'splits': {
        'train': 2000,
        'val': 250,
        'test': 250,
    },
    
    # Problem generation
    'n_operations': 2,                   # number of operations per problem
    'operations': ['+', '-', '*', '/'],  # allowed operations
    'number_range': (5, 50),             # range for random numbers (inclusive)
    
    # Output
    'output_dir': 'data',
    
    # Reproducibility
    'seed': 42,
}

# ============================================================================
# PROBLEM GENERATION
# ============================================================================


def generate_problem(config: dict) -> dict:
    """Generate single arithmetic problem."""
    n_ops = config['n_operations']
    operations = config['operations']
    num_min, num_max = config['number_range']
    
    # Generate numbers (avoid 0 for division safety)
    nums = [random.randint(num_min, num_max) for _ in range(n_ops + 1)]
    ops = [random.choice(operations) for _ in range(n_ops)]
    
    # Build problem string: "num op num op num ..."
    problem_str = str(nums[0])
    for op, num in zip(ops, nums[1:]):
        problem_str += f" {op} {num}"
    
    # Calculate answer with proper order of operations
    try:
        answer = eval(problem_str)
        if isinstance(answer, float):
            answer = round(answer, 2)
    except ZeroDivisionError:
        return generate_problem(config)
    
    return {
        'problem': problem_str,
        'answer': answer,
        'n_ops': n_ops
    }


def generate_dataset(n: int, config: dict) -> list:
    """Generate n unique problems."""
    problems = []
    seen = set()
    
    while len(problems) < n:
        p = generate_problem(config)
        if p['problem'] not in seen:
            seen.add(p['problem'])
            problems.append(p)
    
    return problems


def main():
    config = CONFIG
    random.seed(config['seed'])
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Generating arithmetic datasets...")
    print(f"Operations: {config['operations']}")
    print(f"Number range: {config['number_range']}")
    print(f"Operations per problem: {config['n_operations']}")
    print(f"Seed: {config['seed']}")
    print()
    
    for split_name, n in config['splits'].items():
        dataset = generate_dataset(n, config)
        
        output_path = output_dir / f'{split_name}.json'
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Show stats
        answers = [p['answer'] for p in dataset]
        print(f"{split_name}: {n} problems saved to {output_path}")
        print(f"  Answer range: {min(answers):.2f} to {max(answers):.2f}")
        print()
    
    # Show samples
    print("Sample problems from test set:")
    with open(output_dir / 'test.json') as f:
        test_data = json.load(f)
    for p in test_data[:5]:
        print(f"  {p['problem']} = {p['answer']}")


if __name__ == "__main__":
    main()