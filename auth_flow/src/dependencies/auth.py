"""
FastAPI dependency for extracting and validating the current user from a
Bearer JWT.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.core.security import InvalidTokenError, decode_access_token
from src.models.user_store import User, fake_get_user_by_email

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Decode the bearer token, look the user up, and return it.

    Raises HTTP 401 if the token is invalid/expired or the user no longer
    exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = fake_get_user_by_email(email)
    if user is None:
        raise credentials_exception

    return user
