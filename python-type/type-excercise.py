#Design House record 3 level with TTypedDict
from typing import TypedDict, List, Dict
import pprint
class SmartDevice(TypedDict):
    name: str
    status: bool
    
class Room(TypedDict):
    floor: int
    devices: List[SmartDevice]

class House(TypedDict):
    house_id: str
    location: str
    rooms: Dict[str, Room]

my_house : House = {
    "house_id": "H001",
    "location" : "Da Nang",
    "rooms" : {
        "living_room" : {
            "floor" : 1,
            "devices" : [
                {"name" : "Smart TV", "status" : True},
                {"name" : "Smart Light", "status" : False}
            ]
        }
    }
}
# Func merge_configs

def merge_configs(default: dict, user:dict) -> dict:
    final_config = default.copy()
    
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(final_config.get(key), dict):
            final_config[key] = merge_configs(final_config[key], value)
        else:
            final_config[key] = value
    return final_config

default_app_config = {
    "theme": "dark",
    "language": "en",
    "editor": {
        "font_size": 14,
        "tab_size": 4,
        "plugins": {"git": True, "linters": False}
    }
}

user_custom_config = {
    "language": "vi",  
    "editor": {
        "font_size": 16,  
        "plugins": {"linters": True}  
    }
}
final_sys_config = merge_configs(default_app_config, user_custom_config)
pprint.pprint(final_sys_config)

#Faster Search by Dict Comprehension
raw_users = [
    {"id": "U1", "name": "Hoai", "role": "Admin"},
    {"id": "U2", "name": "Alice", "role": "User"},
    {"id": "U3", "name": "Bob", "role": "Manager"}
]

user_lookup = { user["id"]: { k: v for k, v in user.items() if k != "id"} for user in raw_users }
# pprint.pprint(user_lookup)
print(user_lookup.get('U1'))
# 0(1)


#Type setting from Dict[str, Any] to Field
from typing import Any, Dict
def parse_to_house_record(raw_data: Dict[str, Any]) -> House:
    
    return {
        "house_id": raw_data.get("house_id", ""),
        "location": raw_data.get("location", ""),
        "rooms": raw_data.get("rooms", {})       
    }
# raw_api_case_1 = {}

raw_api_case_2 = {
    "house_id": "H002",
    "location": "Hanoi",
    "rooms": {
        "bedroom": { "floor": 2, "devices": []}
    }
}
# house_1 = parse_to_house_record(raw_api_case_1)
house_2 = parse_to_house_record(raw_api_case_2)
# pprint.pprint(house_1)
pprint.pprint(house_2)