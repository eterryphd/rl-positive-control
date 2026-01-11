import re
from typing import Optional

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