from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1)


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)
