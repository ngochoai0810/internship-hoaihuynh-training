from typing import Any, Dict, List, Tuple, Union

def process(x:Any) -> Any:
    return x.what_ever_i_want()

# union is different from Any, because Any can be any type, but Union is a specific set of types.
# def parse_price(value:Union[str, int, float]) -> float:
#     return float(value)

# can be written like:
def parse_price(value: str | int | float) -> float:
    return float(value)

#Optional[x]
def process_optional(value: str | None) -> str | None:
    return value
 
# y just be a str or None, if no provided value to y, it will be None
def foo(x:int, y:str | None = None) -> None:
    if y is not None:
        print(f"x: {x}, y: {y}")
    else:
        print(f"x: {x}, y is None")


def get_name(user_id : int) -> str| None:
    if user_id == 0:
        return None
    return "Alice"

# def find_house_by_id(houses: list[dict], id: int) -> dict:
#     for house in houses:
#         if house.get("id") == id:
#             return house
#     return None

