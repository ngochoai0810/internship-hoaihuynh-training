"""Password hashing and JWT helpers."""

from datetime import UTC, datetime, timedelta

import jwt
from core.config import Settings
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

PASSWORD_HASH = PasswordHash((BcryptHasher(),))


class InvalidTokenError(Exception):
    """Raised when a JWT cannot be trusted."""


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""

    return PASSWORD_HASH.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return whether a plaintext password matches a stored hash."""

    return PASSWORD_HASH.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    *,
    settings: Settings,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT whose subject is a database user id."""

    lifetime = expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    expires_at = datetime.now(UTC) + lifetime
    return jwt.encode(
        {"sub": subject, "exp": expires_at},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_access_token(token: str, *, settings: Settings) -> str:
    """Validate a JWT and return its non-empty string subject."""

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError("Invalid or expired access token") from exc

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("Access token has no valid subject")
    return subject
