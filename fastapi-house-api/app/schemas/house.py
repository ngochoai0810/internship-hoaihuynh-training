from pydantic import BaseModel, ConfigDict, Field


class HouseBase(BaseModel):
    area: float = Field(gt=0, examples=[120.5])
    rooms: int = Field(gt=0, examples=[3])
    location: str = Field(min_length=1, examples=["District 1"])


class HouseCreate(HouseBase):
    pass


class HouseResponse(HouseBase):
    id: int
    price: float = Field(ge=0, examples=[250000.0])

    model_config = ConfigDict(from_attributes=True)
