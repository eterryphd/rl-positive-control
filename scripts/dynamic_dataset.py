# scripts/dynamic_dataset.py
"""
Dynamic dataset that generates fresh problems on-the-fly.

Instead of pre-generating a fixed dataset, this IterableDataset
generates new problems infinitely. This ensures:
1. Every training step sees a fresh problem
2. No risk of overfitting to a fixed set
3. Effectively infinite training data

Works with distributed training - each rank gets different problems
via rank-based seed offset.
"""

from torch.utils.data import IterableDataset
from typing import Dict, Iterator, Optional
import random

from generators.base import ProblemGenerator
from utils import build_prompt


class DynamicProblemDataset(IterableDataset):
    """
    Infinite dataset that generates fresh problems on-the-fly.
    
    Each call to __iter__ returns a new iterator that generates
    problems forever. The trainer controls how many to use via max_steps.
    
    For distributed training, each rank should use a different seed_offset
    to ensure different problems across ranks.
    """
    
    def __init__(
        self,
        generator: ProblemGenerator,
        tokenizer,
        config: Dict = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            generator: ProblemGenerator instance
            tokenizer: HuggingFace tokenizer for building prompts
            config: Config dict passed to generator
            seed: Base random seed
            rank: Current process rank (for distributed)
            world_size: Total number of processes
        """
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config or {}
        self.system_message = generator.system_message
        
        # Each rank gets a different seed to avoid duplicate problems
        self.seed = seed + rank * 100000
        self.rank = rank
        self.world_size = world_size
        
    def __iter__(self) -> Iterator[Dict]:
        """
        Yield fresh problems forever.
        
        Each problem is guaranteed to be newly generated (though
        collisions are possible given finite problem space).
        """
        # Initialize RNG for this iterator
        rng = random.Random(self.seed)
        
        # Track problems within this iterator to avoid immediate duplicates
        recent_problems = set()
        max_recent = 10000  # Don't let the set grow unbounded
        
        while True:
            # Generate a fresh problem
            # Temporarily set the global random state for the generator
            old_state = random.getstate()
            random.setstate(rng.getstate())
            
            problem = self.generator.generate_problem(self.config)
            
            # Update our RNG state and restore global
            rng.setstate(random.getstate())
            random.setstate(old_state)
            
            # Skip if we've seen this very recently (within same epoch)
            if problem['problem'] in recent_problems:
                continue
            
            recent_problems.add(problem['problem'])
            if len(recent_problems) > max_recent:
                # Remove oldest (approximately - sets don't preserve order)
                recent_problems.pop()
            
            # Build prompt
            prompt = build_prompt(
                problem['problem'],
                self.tokenizer,
                self.system_message
            )
            
            yield {
                'prompt': prompt,
                'problem': problem['problem'],
                'answer': problem['answer'],
            }


class SizedDynamicDataset(DynamicProblemDataset):
    """
    Dynamic dataset with a nominal size.
    
    Some trainers need __len__ even for iterable datasets.
    This provides a nominal length while still generating fresh problems.
    """
    
    def __init__(
        self,
        generator: ProblemGenerator,
        tokenizer,
        config: Dict = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        nominal_size: int = 10000,
    ):
        super().__init__(generator, tokenizer, config, seed, rank, world_size)
        self.nominal_size = nominal_size
    
    def __len__(self) -> int:
        """
        Return nominal size for progress bars and schedulers.
        
        Note: This doesn't limit actual iteration - the dataset
        is still infinite. This is just for trainer compatibility.
        """
        return self.nominal_size
