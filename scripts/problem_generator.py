# scripts/problem_generator.py
"""
Library module for generating arithmetic problems.

This module provides a concrete implementation of the ProblemGenerator
for arithmetic (multiplication) problems.
"""

import json
import random
from typing import Dict, List
from pathlib import Path
from generators.base import ProblemGenerator

# ============================================================================
# CONFIG - Default settings, can be overridden via function arguments
# ============================================================================

DEFAULT_CONFIG = {
    # Problem generation
    'n_operations': 1,                   # Fixed to 1 for multiplication
    'operations': ['*'],                 # Restricted to multiplication
    'min_digits_per_op': {1: [3, 2]},    # First: 3 digits, second: 2 digits
    'max_digits_per_op': {1: [3, 2]},    # Enforce exact digit lengths
    
    # Reproducibility
    'seed': 42,
}

# ============================================================================
# ARITHMETIC GENERATOR
# ============================================================================

class ArithmeticGenerator(ProblemGenerator):
    """Concrete generator for arithmetic multiplication problems."""
    
    def generate_problem(self, config: Dict = None) -> Dict:
        """Generate a single arithmetic problem based on config."""
        if config is None:
            config = DEFAULT_CONFIG
        
        n_ops = config.get('n_operations', DEFAULT_CONFIG['n_operations'])
        operations = config.get('operations', DEFAULT_CONFIG['operations'])
        
        if n_ops != 1 or operations != ['*']:
            raise ValueError("Problem generation restricted to single multiplication.")
        
        min_digits_list = config.get('min_digits_per_op', DEFAULT_CONFIG['min_digits_per_op']).get(n_ops, [3, 2])
        max_digits_list = config.get('max_digits_per_op', DEFAULT_CONFIG['max_digits_per_op']).get(n_ops, [3, 2])
        
        if len(min_digits_list) != n_ops + 1 or len(max_digits_list) != n_ops + 1:
            raise ValueError(f"For {n_ops} operations, must provide {n_ops + 1} min and max digits.")
        
        # Generate numbers within digit constraints
        nums = []
        for min_d, max_d in zip(min_digits_list, max_digits_list):
            if min_d > max_d:
                raise ValueError("Min digits cannot exceed max digits.")
            digits = random.randint(min_d, max_d)
            min_val = 10 ** (digits - 1)
            max_val = (10 ** digits) - 1
            num = random.randint(min_val, max_val)
            nums.append(num)
        
        ops = [random.choice(operations) for _ in range(n_ops)]
        
        # Build problem string: "num op num"
        problem_str = str(nums[0])
        for op, num in zip(ops, nums[1:]):
            problem_str += f" {op} {num}"
        
        # Calculate answer
        try:
            answer = eval(problem_str)
            if isinstance(answer, float):
                answer = round(answer, 2)
        except Exception:
            # Regenerate on error (though unlikely for multiplication)
            return self.generate_problem(config)
        
        return {
            'problem': problem_str,
            'answer': answer,
            'n_ops': n_ops
        }
    
    def generate_batch(self, n: int, config: Dict = None) -> List[Dict]:
        """Generate a batch of n unique problems."""
        if config is None:
            config = DEFAULT_CONFIG
        random.seed(config.get('seed', DEFAULT_CONFIG['seed']))
        
        problems = []
        seen = set()
        
        while len(problems) < n:
            p = self.generate_problem(config)
            if p['problem'] not in seen:
                seen.add(p['problem'])
                problems.append(p)
        
        return problems

# Standalone functions for compatibility
def generate_problem(config: Dict = None) -> Dict:
    return ArithmeticGenerator().generate_problem(config)

def generate_batch(n: int, config: Dict = None) -> List[Dict]:
    return ArithmeticGenerator().generate_batch(n, config)

def generate_dataset(split_name: str, n: int, config: Dict = None, output_dir: str = 'data') -> Path:
    """Generate and save a dataset split (for standalone use if needed)."""
    if config is None:
        config = DEFAULT_CONFIG
    
    dataset = generate_batch(n, config)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f'{split_name}.json'
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return output_path