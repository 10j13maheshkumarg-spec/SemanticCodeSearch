def add_numbers(a, b):
    """Adds two numbers and returns the result."""
    return a + b

def calculate_compound_interest(principal, rate, time):
    """
    Calculates the compound interest.
    """
    amount = principal * (pow((1 + rate / 100), time))
    return amount

class Calculator:
    def __init__(self):
        self.history = []
        
    def multiply(self, x, y):
        result = x * y
        self.history.append(f"{x} * {y} = {result}")
        return result
