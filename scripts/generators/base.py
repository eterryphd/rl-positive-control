# scripts/generators/base.py
"""
Abstract base class for problem generators.

Generators encapsulate ALL task-specific logic:
- Problem generation
- Answer extraction from model output  
- Reward computation
- Correctness checking
- System prompt

This allows train.py and evaluate.py to be completely task-agnostic.
Swapping from arithmetic to interleaving is just changing which generator is used.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Set, Optional


class ProblemGenerator(ABC):
    """
    Abstract base class for problem generators.
    
    To implement a new task:
    1. Subclass ProblemGenerator
    2. Implement all abstract methods/properties
    3. Update CONFIG['generator_class'] in train.py
    
    The generator is the single source of truth for task-specific behavior.
    """
    
    # ========================================================================
    # TASK IDENTITY
    # ========================================================================
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Human-readable task name for logging."""
        pass
    
    @property
    @abstractmethod
    def system_message(self) -> str:
        """System prompt that defines the task for the model."""
        pass
    
    # ========================================================================
    # PROBLEM GENERATION
    # ========================================================================
    
    @abstractmethod
    def generate_problem(self, config: Dict = None) -> Dict:
        """
        Generate a single problem.
        
        Args:
            config: Optional configuration overrides.
        
        Returns:
            Dict with at minimum:
                - 'problem': str - The problem statement (user message)
                - 'answer': Any - The expected answer (type is task-specific)
            
            May include additional metadata (e.g., 'n_ops', 'difficulty').
        """
        pass
    
    @abstractmethod
    def generate_batch(self, n: int, config: Dict = None) -> List[Dict]:
        """
        Generate n unique problems.
        
        Args:
            n: Number of problems to generate.
            config: Optional configuration overrides.
        
        Returns:
            List of problem dicts.
        """
        pass
    
    @abstractmethod
    def generate_held_out(self, n: int, seen: Set[str], config: Dict = None) -> List[Dict]:
        """
        Generate n unique problems NOT in the seen set.
        
        Used for validation to ensure we're testing on unseen problems.
        
        Args:
            n: Number of problems to generate.
            seen: Set of problem strings to exclude.
            config: Optional configuration overrides.
        
        Returns:
            List of problem dicts, none of which have 'problem' in seen.
        """
        pass
    
    # ========================================================================
    # ANSWER EXTRACTION & EVALUATION
    # ========================================================================
    
    @abstractmethod
    def extract_answer(self, response: str) -> Optional[Any]:
        """
        Extract the answer from model's response.
        
        Args:
            response: Raw model output string.
        
        Returns:
            Extracted answer in task-appropriate type, or None if extraction fails.
            
        Examples:
            - Arithmetic: float (e.g., 42.0)
            - Interleaving: str (the interleaved sequence)
        """
        pass
    
    @abstractmethod
    def compute_reward(self, predicted: Any, expected: Any) -> float:
        """
        Compute reward for a prediction.
        
        Args:
            predicted: Extracted answer from model (may be None).
            expected: Ground truth answer from problem dict.
        
        Returns:
            Reward in [-1.0, 1.0] range.
            - 1.0 = perfect
            - 0.0 = neutral
            - -1.0 = complete failure or invalid output
        """
        pass
    
    @abstractmethod
    def check_correct(self, predicted: Any, expected: Any) -> bool:
        """
        Binary correctness check.
        
        Used for accuracy metrics in validation/evaluation.
        
        Args:
            predicted: Extracted answer from model (may be None).
            expected: Ground truth answer from problem dict.
        
        Returns:
            True if answer is correct, False otherwise.
        """
        pass
    
    # ========================================================================
    # OPTIONAL: LOGGING & DEBUGGING
    # ========================================================================
    
    def format_example(self, problem: Dict, response: str, predicted: Any) -> str:
        """
        Format a single example for logging/debugging.
        
        Override for task-specific formatting.
        
        Args:
            problem: The problem dict.
            response: Raw model response.
            predicted: Extracted answer.
        
        Returns:
            Human-readable string for logging.
        """
        correct = self.check_correct(predicted, problem['answer'])
        status = "✓" if correct else "✗"
        return (
            f"{status} Problem: {problem['problem']}\n"
            f"  Expected: {problem['answer']}\n"
            f"  Got: {predicted}\n"
            f"  Raw: '{response[:100]}{'...' if len(response) > 100 else ''}'"
        )