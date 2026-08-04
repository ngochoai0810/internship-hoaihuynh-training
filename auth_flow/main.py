"""
Small FastAPI demo app for testing the auth flow through Swagger UI.
"""

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.core.security import create_access_token, hash_password, verify_password
from src.dependencies.auth import get_current_user
from src.models.user_store import User, add_user, fake_get_user_by_email

app = FastAPI(title="Auth Flow Demo")

DEMO_EMAIL = "admin@example.com"
DEMO_PASSWORD = "Admin123!"


def seed_demo_user() -> None:
    """Create one demo user for manual Swagger testing."""
    if fake_get_user_by_email(DEMO_EMAIL) is None:
        add_user(
            User(
                id=1,
                email=DEMO_EMAIL,
                hashed_password=hash_password(DEMO_PASSWORD),
            )
        )


seed_demo_user()


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    user = fake_get_user_by_email(form_data.username)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token({"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
def read_current_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user
