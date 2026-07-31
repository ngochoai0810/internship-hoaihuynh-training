
def parse_price(value : str | int | float) -> float:
    #rule: accept int, float, str input
    # if isinstance(value, str):
    #     return float(value)
    # return float(value)
    if value is None:
        raise ValueError("Value cannot be None")
    
    if isinstance(value, str):
        value = value.strip()
        
        if value == "":
            raise ValueError("Value cannot be an empty string")
        value = value.replace(",", "")
    
    try:
        result = float(value)
    except(TypeError, ValueError):
        raise ValueError(f"Invalid price format: {value}")
    
    if result < 0:
        raise ValueError("Price cannot be negative")
    
    return result

print(parse_price(1000))
print(parse_price(99.5))
print(parse_price("25000"))
print(parse_price("330,000"))
print(parse_price("  4500  "))
try:
    print(parse_price(""))
except ValueError as e:
    print(e)
try:
    print(parse_price("abc"))
except ValueError as e:
    print(e)
try:
    print(parse_price(-100))
except ValueError as e:
    print(e)
