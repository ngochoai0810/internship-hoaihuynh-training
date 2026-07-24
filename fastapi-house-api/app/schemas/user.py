from pydantic import BaseModel, ConfigDict, Field


class UserBase(BaseModel):
    email: str = Field(min_length=3)
    full_name: str = Field(min_length=1)


class UserCreate(UserBase):
    password: str = Field(min_length=8)

class UserResponse(UserBase):
    id: int
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)
