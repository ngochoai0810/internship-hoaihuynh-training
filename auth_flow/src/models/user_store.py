"""
Minimal in-memory user store.

This stands in for the real SQLAlchemy User model / DB session so
`get_current_user` can be tested independently.
"""

from dataclasses import dataclass


@dataclass
class User:
    id: int
    email: str
    hashed_password: str
    is_active: bool = True


_FAKE_USERS_DB: dict[str, User] = {}


def add_user(user: User) -> None:
    _FAKE_USERS_DB[user.email] = user


def fake_get_user_by_email(email: str) -> User | None:
    return _FAKE_USERS_DB.get(email)
