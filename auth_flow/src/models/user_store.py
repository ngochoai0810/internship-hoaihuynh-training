"""
Minimal in-memory user store.

This stands in for the real SQLAlchemy User model / DB session. In a real
database, duplicate email handling would come from a unique constraint and an
IntegrityError; here we mirror that behavior with DuplicateEmailError.
"""

from dataclasses import dataclass


@dataclass
class User:
    id: int
    email: str
    hashed_password: str
    is_active: bool = True


class DuplicateEmailError(Exception):
    """Raised when an email is already present in the fake user table."""


_FAKE_USERS_DB: dict[str, User] = {}
_NEXT_ID = 1


def add_user(user: User) -> None:
    global _NEXT_ID

    _FAKE_USERS_DB[user.email] = user
    _NEXT_ID = max(_NEXT_ID, user.id + 1)


def create_user(email: str, hashed_password: str) -> User:
    """Insert a new user, raising DuplicateEmailError for duplicate emails."""
    global _NEXT_ID

    if email in _FAKE_USERS_DB:
        raise DuplicateEmailError(f"Email '{email}' is already registered")

    user = User(id=_NEXT_ID, email=email, hashed_password=hashed_password)
    _NEXT_ID += 1
    add_user(user)
    return user


def fake_get_user_by_email(email: str) -> User | None:
    return _FAKE_USERS_DB.get(email)


def update_user(
    email: str,
    *,
    new_email: str | None = None,
    new_hashed_password: str | None = None,
) -> User:
    """Update a user by email, raising DuplicateEmailError for email conflicts."""
    user = _FAKE_USERS_DB.pop(email)

    if new_email is not None and new_email != email and new_email in _FAKE_USERS_DB:
        _FAKE_USERS_DB[email] = user
        raise DuplicateEmailError(f"Email '{new_email}' is already registered")

    if new_email is not None:
        user.email = new_email
    if new_hashed_password is not None:
        user.hashed_password = new_hashed_password

    _FAKE_USERS_DB[user.email] = user
    return user


def reset_store() -> None:
    """Clear fake users between tests."""
    global _NEXT_ID

    _FAKE_USERS_DB.clear()
    _NEXT_ID = 1
