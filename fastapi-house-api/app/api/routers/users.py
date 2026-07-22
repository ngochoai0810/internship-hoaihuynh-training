from fastapi import APIRouter, Depends, status

from app.api.deps import get_current_mock_user_id
from app.schemas.user import UserCreate, UserResponse


router = APIRouter(prefix="/users", tags=["users"])

MOCK_USER = {
    "id": 1,
    "email": "intern@example.com",
    "full_name": "Hoai Huynh",
    "is_active": True,
    "hashed_password": "hashed_demo_password",
}


def fake_hash_password(password: str) -> str:
    return f"hashed_{password}"


@router.get("/me", response_model=UserResponse)
def get_current_user(
    user_id: int = Depends(get_current_mock_user_id),
) -> dict[str, object]:
    return {**MOCK_USER, "id": user_id}


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate) -> dict[str, object]:
    return {
        "id": 2,
        "email": payload.email,
        "full_name": payload.full_name,
        "is_active": True,
        "hashed_password": fake_hash_password(payload.password),
    }
