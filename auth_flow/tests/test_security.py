"""
Tests covering password hashing and JWT helpers.
"""

import jwt
import pytest

from src.core.config import settings
from src.core.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_verify_password_correct_and_incorrect():
    hashed = hash_password("MySecret123!")
    assert verify_password("MySecret123!", hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_hash_is_salted_and_differs_each_time():
    h1 = hash_password("SamePassword")
    h2 = hash_password("SamePassword")
    assert h1 != h2
    assert verify_password("SamePassword", h1)
    assert verify_password("SamePassword", h2)


def test_create_and_decode_access_token_roundtrip():
    token = create_access_token({"sub": "user@example.com"})
    payload = decode_access_token(token)
    assert payload["sub"] == "user@example.com"
    assert "exp" in payload


def test_decode_expired_token_raises():
    token = create_access_token({"sub": "user@example.com"}, expires_minutes=-1)
    with pytest.raises(InvalidTokenError):
        decode_access_token(token)


def test_decode_tampered_token_raises():
    token = create_access_token({"sub": "user@example.com"})
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(InvalidTokenError):
        decode_access_token(tampered)


def test_decode_token_signed_with_wrong_key_raises():
    bad_token = jwt.encode(
        {"sub": "user@example.com"},
        "wrong-secret-at-least-32-bytes-long",
        algorithm=settings.ALGORITHM,
    )
    with pytest.raises(InvalidTokenError):
        decode_access_token(bad_token)
