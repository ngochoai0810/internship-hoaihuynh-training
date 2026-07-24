from typing import Any

def add(a:int, b:int) -> int:
    return a + b

def greet(name:str) -> str:
    return f"Hello, {name}"
print (greet(123))

def multiply(a, b) -> float:
    return a * b

print(multiply(2.5, 4.0))

def get_name() -> str:
    return "Alice"

def is_even(number:int) -> bool:
    return number % 2 == 0
print(is_even(10))

def print_value(value:Any) -> None:
    print(value)

print_value("Hello, World!")
print_value(42)
print_value([1, 2, 3])