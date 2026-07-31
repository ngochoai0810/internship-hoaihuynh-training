# Func simulate validate logic for house record
def validate_house_record(record: dict) -> str | None:
    existing_ids = {1, 2, 3}

    if "id" not in record:
        return "Missing 'id' field"

    if "price" not in record:
        return "Missing 'price' field"

    if not isinstance(record["id"], int) or isinstance(record["id"], bool):
        return "Field 'id' must be an integer"

    if record["id"] in existing_ids:
        return "ID already exists"

    price = record["price"]
    
    if not isinstance(price, (int, float, str)) or isinstance(price, bool):
        return "Field 'price' must be a number or string"
    
    if isinstance(price, str):
        price = price.strip()
        if price == "":
            return "Field 'price' cannot be an empty string"
        price = price.replace(",", "")
        
        try:
            price = float(price)
        except ValueError:
            return "Field 'price' must be a valid number"
        
        if price < 0:
            return "Field 'price' cannot be negative"
        return None

# Test cases
print(validate_house_record({
    "id": 10,
    "price": 250000
}))

print(validate_house_record({
    "id": 10
}))

print(validate_house_record({
    "id": 10,
    "price": []
}))