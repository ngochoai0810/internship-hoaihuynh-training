"""
Core security utilities: password hashing and JWT handling.

Day 26 -> password hashing with pwdlib (bcrypt algorithm).
Day 27 -> JWT creation and verification (PyJWT).
"""

from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

from src.core.config import settings

# The plan requires bcrypt specifically, so this pins pwdlib to BcryptHasher
# instead of using PasswordHash.recommended(), which may prefer Argon2.
password_hasher = PasswordHash((BcryptHasher(),))


def hash_password(password: str) -> str:
    """Hash a plaintext password. Never store or log the plaintext value."""
    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True only if `plain_password` matches the given hash."""
    return password_hasher.verify(plain_password, hashed_password)


class InvalidTokenError(Exception):
    """Raised when a token is malformed, expired, or has a bad signature."""


def create_access_token(data: dict, expires_minutes: int | None = None) -> str:
    """
    Build a signed JWT.

    `data` should contain `sub` (subject, e.g. user email/id). `exp` is added
    automatically and the input dictionary is not mutated.
    """
    to_encode = data.copy()
    expire_delta = timedelta(
        minutes=(
            expires_minutes
            if expires_minutes is not None
            else settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )
    expire_at = datetime.now(timezone.utc) + expire_delta
    to_encode.update({"exp": expire_at})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT.

    Raises InvalidTokenError if the token is expired, malformed, or the
    signature does not match.
    """
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError(str(exc)) from exc
