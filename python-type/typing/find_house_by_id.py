
houses = [
    {"id" : 1, "name": "House A"},
    {"id" : 2, "name": "House B"},
]

def find_house_by_id( id : int) -> dict | None:
    for house in houses:
        if house.get("id") == id:
            return house
    return None


print(find_house_by_id(1))
print(find_house_by_id(3))