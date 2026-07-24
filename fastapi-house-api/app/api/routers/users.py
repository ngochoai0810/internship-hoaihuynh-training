from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_mock_user_id
from app.schemas.user import UserCreate, UserResponse


router = APIRouter(prefix="/users", tags=["users"])

MOCK_USERS = {
    1: {
        "id": 1,
        "email": "intern@example.com",
        "full_name": "Hoai Huynh",
        "is_active": True,
        "hashed_password": "hashed_demo_password",
    },
}


def fake_hash_password(password: str) -> str:
    return f"hashed_{password}"


@router.get("", response_model=list[UserResponse])
def list_users() -> list[dict[str, object]]:
    return list(MOCK_USERS.values())


@router.get("/me", response_model=UserResponse)
def get_current_user(
    user_id: int = Depends(get_current_mock_user_id),
) -> dict[str, object]:
    user = MOCK_USERS.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int) -> dict[str, object]:
    user = MOCK_USERS.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate) -> dict[str, object]:
    return {
        "id": 2,
        "email": payload.email,
        "full_name": payload.full_name,
        "is_active": True,
        "hashed_password": fake_hash_password(payload.password),
    }
