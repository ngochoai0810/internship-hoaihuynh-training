import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.schemas.house_schema import HouseBase


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


section("1) Valid HouseBase instance")

house = HouseBase(
    gr_liv_area=1710.0,
    overall_qual=7,
    tot_rms_abv_grd=8,
    neighborhood="  CollgCr  ",
    year_built=2003,
    year_remodeled=2005,
    garage_cars=2,
)
print(house)
print("neighborhood after strip:", repr(house.neighborhood))
print("model_dump() ->")
print(house.model_dump())


section("2) Invalid data -> negative gr_liv_area")

try:
    HouseBase(
        gr_liv_area=-500.0,
        overall_qual=7,
        tot_rms_abv_grd=8,
        neighborhood="CollgCr",
        year_built=2003,
    )
except ValidationError as error:
    print(f"Caught ValidationError with {error.error_count()} error(s):")
    for item in error.errors():
        print(f"  - field: {item['loc']}, msg: {item['msg']}")


section("3) Invalid data -> overall_qual out of range")

try:
    HouseBase(
        gr_liv_area=1200.0,
        overall_qual=15,
        tot_rms_abv_grd=6,
        neighborhood="NAmes",
        year_built=1990,
    )
except ValidationError as error:
    print(f"Caught ValidationError with {error.error_count()} error(s):")
    for item in error.errors():
        print(f"  - field: {item['loc']}, msg: {item['msg']}")


section("4) Invalid data -> year_built in the future")

try:
    HouseBase(
        gr_liv_area=1400.0,
        overall_qual=6,
        tot_rms_abv_grd=7,
        neighborhood="Somerst",
        year_built=2099,
    )
except ValidationError as error:
    print(f"Caught ValidationError with {error.error_count()} error(s):")
    for item in error.errors():
        print(f"  - field: {item['loc']}, msg: {item['msg']}")


section("5) model_validate() from a raw dict")

raw_row = {
    "gr_liv_area": 2198.0,
    "overall_qual": 8,
    "tot_rms_abv_grd": 9,
    "neighborhood": "NridgHt",
    "year_built": 2001,
    "year_remodeled": None,
    "garage_cars": 3,
}

house_from_dict = HouseBase.model_validate(raw_row)
print(house_from_dict)
print(
    "Round-trip equal to dumped dict?",
    house_from_dict.model_dump()["neighborhood"] == raw_row["neighborhood"],
)
