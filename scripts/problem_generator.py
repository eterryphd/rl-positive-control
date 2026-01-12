# scripts/problem_generator.py
"""
Arithmetic problem generator for positive control experiments.

Generates 3-digit × 2-digit multiplication problems with hybrid rewards.
Uses min(NW_alignment, relative_error) to penalize both pattern and magnitude errors.
This validates the RL pipeline before applying to interleaving.
"""

import math
import random
import re
from typing import Dict, List, Set, Optional
from generators.base import ProblemGenerator


def needleman_wunsch_identity(seq1: list, seq2: list, 
                               match: int = 1, 
                               mismatch: int = -1, 
                               gap: int = -2) -> float:
    """
    Needleman-Wunsch global alignment, returns identity score [0, 1].
    
    Args:
        seq1, seq2: Sequences to align (lists of comparable tokens)
        match: Score for matching tokens
        mismatch: Score for mismatching tokens  
        gap: Score for gap (insertion/deletion)
    
    Returns:
        Identity as fraction of matches in optimal alignment
    """
    n, m = len(seq1), len(seq2)
    
    # Handle empty sequences
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    
    # Initialize DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Base cases: gaps along edges
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                diag = dp[i-1][j-1] + match
            else:
                diag = dp[i-1][j-1] + mismatch
            
            up = dp[i-1][j] + gap
            left = dp[i][j-1] + gap
            
            dp[i][j] = max(diag, up, left)
    
    # Traceback to count matches
    i, j = n, m
    matches = 0
    alignment_length = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                score_diag = dp[i-1][j-1] + match
            else:
                score_diag = dp[i-1][j-1] + mismatch
        else:
            score_diag = float('-inf')
        
        score_up = dp[i-1][j] + gap if i > 0 else float('-inf')
        score_left = dp[i][j-1] + gap if j > 0 else float('-inf')
        
        if score_diag >= score_up and score_diag >= score_left and i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                matches += 1
            i -= 1
            j -= 1
        elif score_up >= score_left and i > 0:
            i -= 1
        else:
            j -= 1
        
        alignment_length += 1
    
    return matches / alignment_length if alignment_length > 0 else 0.0


class ArithmeticGenerator(ProblemGenerator):
    """
    Generator for arithmetic (multiplication) problems.
    
    Problem format: "XXX * YY" where XXX is 3 digits, YY is 2 digits.
    Answer: integer product
    
    Uses hybrid reward: min(NW_alignment, relative_error) to penalize
    both pattern errors and magnitude errors.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'n_operations': 1,
        'operations': ['*'],
        'min_digits_per_op': {1: [3, 2]},  # [first_operand, second_operand] - 3x2 digits
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
        Compute reward using min(NW_alignment, relative_error).
        
        This penalizes both:
        - Pattern errors (wrong digits) via Needleman-Wunsch
        - Magnitude errors (wrong scale) via relative error
        
        Must be good on BOTH metrics to get high reward.
        
        Returns:
            Reward in [-1.0, 1.0]
        """
        if predicted is None:
            return -1.0
        
        # Convert to digit sequences for NW alignment
        pred_str = str(int(abs(predicted)))
        exp_str = str(int(abs(expected)))
        
        pred_digits = list(pred_str)
        exp_digits = list(exp_str)
        
        # NW identity [0, 1] -> reward [-1, 1]
        nw_identity = needleman_wunsch_identity(pred_digits, exp_digits)
        nw_reward = 2 * nw_identity - 1
        
        # Relative error -> reward [-1, 1]
        relative_error = abs(predicted - expected) / (abs(expected) + 1)
        raw_relative = math.exp(-relative_error * self.REWARD_SCALE)
        relative_reward = 2 * raw_relative - 1
        
        # Take minimum: must be good on BOTH to score high
        return min(nw_reward, relative_reward)
    
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