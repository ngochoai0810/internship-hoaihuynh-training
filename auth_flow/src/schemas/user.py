"""
Auth-related Pydantic schemas.

Follows the Schema Segregation pattern from Day 18: separate
Create/Response/Update schemas so responses never leak password fields.
"""

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Input for POST /register."""

    email: EmailStr
    password: str = Field(min_length=8)


class UserResponse(BaseModel):
    """Output for auth routes. Never includes the password/hash."""

    id: int
    email: EmailStr
    is_active: bool

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    """Partial update schema for PATCH /me."""

    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=8)


class Token(BaseModel):
    """Response body for POST /login."""

    access_token: str
    token_type: str = "bearer"
