# scripts/utils.py
import re
from typing import Optional

# Centralized system message
SYSTEM_MESSAGE = "You are a calculator. Output only the number."

def extract_answer(response: str) -> Optional[float]:
    """
    Extracts the first numeric value from the model response, handling formatting issues like commas.
    
    Args:
        response (str): The raw model output.
    
    Returns:
        Optional[float]: The extracted number as a float, or None if no valid number is found.
    """
    # Remove commas to handle thousands separators
    response = response.replace(',', '')
    
    # Find the first number (supports negatives and decimals)
    numbers = re.findall(r'-?\d+\.?\d*', response)
    
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    
    return None

def build_prompt(problem: str, tokenizer) -> str:
    """
    Builds a chat-formatted prompt for the given problem using the centralized system message.
    
    Args:
        problem (str): The arithmetic problem string.
        tokenizer: The tokenizer object for applying chat template.
    
    Returns:
        str: The formatted prompt.
    """
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": problem}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )