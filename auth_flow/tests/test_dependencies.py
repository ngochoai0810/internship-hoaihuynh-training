"""
Tests for the get_current_user dependency.
"""

import pytest
from fastapi import HTTPException

from src.core.security import create_access_token
from src.dependencies.auth import get_current_user
from src.models.user_store import User, add_user


def setup_module():
    add_user(User(id=1, email="known@example.com", hashed_password="x"))


def test_get_current_user_with_valid_token_returns_user():
    token = create_access_token({"sub": "known@example.com"})
    user = get_current_user(token=token)
    assert user.email == "known@example.com"


def test_get_current_user_with_invalid_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token="not-a-real-token")
    assert exc_info.value.status_code == 401


def test_get_current_user_for_unknown_email_raises_401():
    token = create_access_token({"sub": "ghost@example.com"})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token=token)
    assert exc_info.value.status_code == 401
