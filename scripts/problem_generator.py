# scripts/problem_generator.py
"""
Arithmetic problem generator for positive control experiments.

Generates 3-digit × 2-digit multiplication problems.
This validates the RL pipeline before applying to interleaving.
"""

import math
import random
import re
from typing import Dict, List, Set, Optional
from generators.base import ProblemGenerator


class ArithmeticGenerator(ProblemGenerator):
    """
    Generator for arithmetic (multiplication) problems.
    
    Problem format: "XXX * YY" where XXX is 3 digits, YY is 2 digits.
    Answer: integer product
    
    This is a positive control - simple enough that RL should definitely
    improve performance, validating the training pipeline works.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'n_operations': 1,
        'operations': ['*'],
        'min_digits_per_op': {1: [3, 2]},  # [first_operand, second_operand]
        'max_digits_per_op': {1: [3, 2]},
        'seed': 42,
    }
    
    # Evaluation parameters
    TOLERANCE = 0.01  # For floating point comparison
    REWARD_SCALE = 5  # Exponential decay rate for partial credit
    
    # ========================================================================
    # TASK IDENTITY
    # ========================================================================
    
    @property
    def task_name(self) -> str:
        return "arithmetic-multiplication"
    
    @property
    def system_message(self) -> str:
        return "You are a calculator. Output only the number."
    
    # ========================================================================
    # PROBLEM GENERATION
    # ========================================================================
    
    def generate_problem(self, config: Dict = None) -> Dict:
        """Generate a single multiplication problem."""
        config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        n_ops = config['n_operations']
        operations = config['operations']
        
        if n_ops != 1 or operations != ['*']:
            raise ValueError("ArithmeticGenerator restricted to single multiplication.")
        
        min_digits = config['min_digits_per_op'][n_ops]
        max_digits = config['max_digits_per_op'][n_ops]
        
        # Generate operands with specified digit counts
        nums = []
        for min_d, max_d in zip(min_digits, max_digits):
            digits = random.randint(min_d, max_d)
            min_val = 10 ** (digits - 1)
            max_val = (10 ** digits) - 1
            nums.append(random.randint(min_val, max_val))
        
        problem_str = f"{nums[0]} * {nums[1]}"
        answer = nums[0] * nums[1]
        
        return {
            'problem': problem_str,
            'answer': answer,
            'n_ops': n_ops,
        }
    
    def generate_batch(self, n: int, config: Dict = None) -> List[Dict]:
        """Generate n unique multiplication problems."""
        config = {**self.DEFAULT_CONFIG, **(config or {})}
        random.seed(config.get('seed', 42))
        
        problems = []
        seen = set()
        
        while len(problems) < n:
            p = self.generate_problem(config)
            if p['problem'] not in seen:
                seen.add(p['problem'])
                problems.append(p)
        
        return problems
    
    def generate_held_out(self, n: int, seen: Set[str], config: Dict = None) -> List[Dict]:
        """Generate n problems not in the seen set."""
        config = {**self.DEFAULT_CONFIG, **(config or {})}
        # Use different seed for held-out to avoid overlap
        config['seed'] = config.get('seed', 42) + 10000
        random.seed(config['seed'])
        
        problems = []
        attempts = 0
        max_attempts = n * 100  # Prevent infinite loop
        
        while len(problems) < n and attempts < max_attempts:
            p = self.generate_problem(config)
            if p['problem'] not in seen:
                seen.add(p['problem'])  # Also add to seen to avoid duplicates within held-out
                problems.append(p)
            attempts += 1
        
        if len(problems) < n:
            print(f"    Warning: Could only generate {len(problems)}/{n} held-out problems")
        
        return problems
    
    # ========================================================================
    # ANSWER EXTRACTION & EVALUATION
    # ========================================================================
    
    def extract_answer(self, response: str) -> Optional[float]:
        """
        Extract numeric answer from model response.
        
        Handles:
        - Comma-separated numbers (1,234 -> 1234)
        - Negative numbers
        - Decimals
        - Numbers embedded in text
        """
        # Remove commas (thousands separators)
        response = response.replace(',', '')
        
        # Find first number (supports negatives and decimals)
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        
        return None
    
    def compute_reward(self, predicted: Optional[float], expected: int) -> float:
        """
        Compute reward based on relative error.
        
        Uses exponential decay:
        - Perfect answer: +1.0
        - 10% off: ~0.2
        - 50% off: ~-0.9
        - Failed extraction: -1.0
        
        Returns:
            Reward in [-1.0, 1.0]
        """
        if predicted is None:
            return -1.0
        
        # Relative error (add 1 to denominator to handle answer=0)
        relative_error = abs(predicted - expected) / (abs(expected) + 1)
        
        # Exponential decay
        raw_reward = math.exp(-relative_error * self.REWARD_SCALE)
        
        # Scale to [-1, 1]
        return 2 * raw_reward - 1
    
    def check_correct(self, predicted: Optional[float], expected: int) -> bool:
        """Check if prediction matches expected within tolerance."""
        if predicted is None:
            return False
        return abs(predicted - expected) < self.TOLERANCE
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def format_example(self, problem: Dict, response: str, predicted: Optional[float]) -> str:
        """Format for logging with reward info."""
        correct = self.check_correct(predicted, problem['answer'])
        reward = self.compute_reward(predicted, problem['answer'])
        status = "✓" if correct else "✗"
        
        return (
            f"{status} {problem['problem']} = {problem['answer']}\n"
            f"   Predicted: {predicted} (reward: {reward:.3f})\n"
            f"   Raw: '{response[:80]}{'...' if len(response) > 80 else ''}'"
        )


# =============================================================================
# STANDALONE FUNCTIONS (backward compatibility)
# =============================================================================

def generate_problem(config: Dict = None) -> Dict:
    """Standalone function for backward compatibility."""
    return ArithmeticGenerator().generate_problem(config)


def generate_batch(n: int, config: Dict = None) -> List[Dict]:
    """Standalone function for backward compatibility."""
    return ArithmeticGenerator().generate_batch(n, config)