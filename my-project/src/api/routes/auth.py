"""Database-backed registration and login routes."""

from typing import Annotated

from core.security import create_access_token, hash_password, verify_password
from database import get_db
from dependencies.auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from models import User
from schemas.auth import TokenResponse, UserCreate, UserResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: UserCreate,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Create a database-backed user with a hashed password."""

    user = User(
        email=str(payload.email).lower(),
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc
    return user


@router.post("/login", response_model=TokenResponse)
def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Authenticate an email/password pair and issue a Bearer JWT."""

    user = db.scalar(select(User).where(User.email == form_data.username.lower()))
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        str(user.id),
        settings=request.app.state.settings,
    )
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def read_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Return the public identity encoded by the Bearer token."""

    return current_user
