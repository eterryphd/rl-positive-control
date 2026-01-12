# scripts/utils.py
"""
Shared utilities for RL training pipeline.

This module contains ONLY task-agnostic utilities.
All task-specific logic (answer extraction, rewards, system messages)
lives in the generator classes.
"""

from typing import Any


def build_prompt(problem: str, tokenizer, system_message: str) -> str:
    """
    Build a chat-formatted prompt for the given problem.
    
    Args:
        problem: The problem string (becomes user message).
        tokenizer: HuggingFace tokenizer with chat template support.
        system_message: Task-specific system prompt (from generator).
    
    Returns:
        Formatted prompt string ready for model input.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": problem}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def truncate_string(s: str, max_len: int = 100, suffix: str = "...") -> str:
    """Truncate string with suffix if too long."""
    if len(s) <= max_len:
        return s
    return s[:max_len - len(suffix)] + suffix