from abc import ABC, abstractmethod
from typing import Dict, List

class ProblemGenerator(ABC):
    """
    Abstract base class for problem generators.
    
    This defines the interface for generating problems, allowing substitution
    between different experiment types (e.g., arithmetic, interleave) via configuration.
    """
    
    @abstractmethod
    def generate_problem(self, config: Dict = None) -> Dict:
        """
        Generate a single problem based on the provided config.
        
        Args:
            config (Dict, optional): Configuration overrides.
        
        Returns:
            Dict: Problem dictionary with 'problem', 'answer', and metadata.
        """
        pass
    
    @abstractmethod
    def generate_batch(self, n: int, config: Dict = None) -> List[Dict]:
        """
        Generate a batch of n unique problems.
        
        Args:
            n (int): Number of problems to generate.
            config (Dict, optional): Configuration overrides.
        
        Returns:
            List[Dict]: List of problem dictionaries.
        """
        pass