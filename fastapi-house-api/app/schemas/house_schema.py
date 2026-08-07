"""
Concepts covered:
- BaseModel basics: fields, types, defaults
- Field() constraints
- Optional fields with int | None
- ConfigDict(str_strip_whitespace=True)
- field_validator for business-rule validation
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HouseBase(BaseModel):
    """
    Mirrors selected Kaggle House Prices columns with explicit types.

    No Any type is used in this schema.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    gr_liv_area: float = Field(gt=0)
    overall_qual: int = Field(ge=0, le=10)
    tot_rms_abv_grd: int = Field(ge=1)
    neighborhood: str = Field(min_length=1)
    year_built: int
    year_remodeled: int | None = None
    garage_cars: int | None = None

    @field_validator("year_built")
    @classmethod
    def year_built_not_in_future(cls, value: int) -> int:
        current_year = date.today().year
        if value > current_year:
            raise ValueError(
                f"year_built ({value}) cannot be later than the current year "
                f"({current_year})"
            )
        return value

    @field_validator("year_remodeled")
    @classmethod
    def year_remodeled_not_in_future(cls, value: int | None) -> int | None:
        if value is not None and value > date.today().year:
            raise ValueError("year_remodeled cannot be in the future")
        return value
    