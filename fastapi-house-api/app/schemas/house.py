from pydantic import BaseModel, ConfigDict, Field


class HouseBase(BaseModel):
    area: float = Field(gt=0, examples=[120.5])
    rooms: int = Field(gt=0, examples=[3])
    location: str = Field(min_length=1, examples=["District 1"])


class HouseCreate(HouseBase):
    pass


class HouseUpdate(BaseModel):
    area: float | None = Field(default=None, gt=0, examples=[120.5])
    rooms: int | None = Field(default=None, gt=0, examples=[3])
    location: str | None = Field(default=None, min_length=1, examples=["District 1"])
    price: float | None = Field(default=None, ge=0, examples=[250000.0])


class HouseResponse(HouseBase):
    id: int
    price: float = Field(ge=0, examples=[250000.0])

    model_config = ConfigDict(from_attributes=True)
