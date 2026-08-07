"""Day 28 auth routes: register, login, and protected user profile."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.core.security import create_access_token, hash_password, verify_password
from src.dependencies.auth import get_current_user
from src.models.user_store import (
    DuplicateEmailError,
    User,
    create_user,
    fake_get_user_by_email,
    update_user,
)
from src.schemas.user import Token, UserCreate, UserResponse, UserUpdate

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(payload: UserCreate) -> User:
    """Create a new user, rejecting duplicate email addresses."""
    try:
        return create_user(
            email=str(payload.email),
            hashed_password=hash_password(payload.password),
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        ) from exc


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    """Authenticate with OAuth2 form fields and return a bearer token."""
    user = fake_get_user_by_email(form_data.username)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token({"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def read_me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user."""
    return current_user


@router.patch("/me", response_model=UserResponse)
def update_me(
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
) -> User:
    """Partially update the current user's email and/or password."""
    new_hashed_password = (
        hash_password(payload.password) if payload.password is not None else None
    )
    try:
        return update_user(
            current_user.email,
            new_email=str(payload.email) if payload.email is not None else None,
            new_hashed_password=new_hashed_password,
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        ) from exc
