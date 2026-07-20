from fastapi import APIRouter, Depends, status

from app.api.deps import get_current_mock_user_id
from app.schemas.user import UserCreate, UserResponse


router = APIRouter(prefix="/users", tags=["users"])

MOCK_USER = UserResponse(
    id=1,
    email="intern@example.com",
    full_name="Hoai Huynh",
    is_active=True,
)


@router.get("/me", response_model=UserResponse)
def get_current_user(
    user_id: int = Depends(get_current_mock_user_id),
) -> UserResponse:
    return MOCK_USER.model_copy(update={"id": user_id})


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate) -> UserResponse:
    return UserResponse(
        id=2,
        email=payload.email,
        full_name=payload.full_name,
        is_active=True,
    )
