# scripts/generators/__init__.py
"""
Problem generators for RL training experiments.

Each generator encapsulates all task-specific logic, making the training
pipeline completely task-agnostic.

Available generators:
- ArithmeticGenerator: Positive control (3-digit × 2-digit multiplication)
- (Future) InterleaveGenerator: Main experiment task
"""

from .base import ProblemGenerator

__all__ = ['ProblemGenerator']