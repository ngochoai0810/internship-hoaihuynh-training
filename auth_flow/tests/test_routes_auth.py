"""End-to-end tests for the Day 28 auth routes."""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.user_store import reset_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_store():
    reset_store()
    yield
    reset_store()


def _register(email: str = "user@example.com", password: str = "StrongPass123"):
    return client.post("/register", json={"email": email, "password": password})


def _login(email: str = "user@example.com", password: str = "StrongPass123"):
    return client.post("/login", data={"username": email, "password": password})


def test_register_returns_201_and_never_leaks_password():
    response = _register()

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 1
    assert body["email"] == "user@example.com"
    assert body["is_active"] is True
    assert "password" not in body
    assert "hashed_password" not in body


def test_register_duplicate_email_returns_400():
    _register()

    response = _register()

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


def test_register_invalid_email_or_short_password_returns_422():
    invalid_email = _register(email="not-an-email")
    short_password = _register(password="short")

    assert invalid_email.status_code == 422
    assert short_password.status_code == 422


def test_login_success_returns_bearer_token():
    _register()

    response = _login()

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
    assert len(body["access_token"]) > 0


def test_login_wrong_password_returns_401():
    _register()

    response = _login(password="WrongPassword")

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_login_unknown_email_returns_401():
    response = _login(email="ghost@example.com")

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_me_without_token_returns_401():
    response = client.get("/me")

    assert response.status_code == 401


def test_me_with_valid_token_returns_current_user():
    _register()
    token = _login().json()["access_token"]

    response = client.get("/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "user@example.com"
    assert body["is_active"] is True
    assert "hashed_password" not in body


def test_me_with_garbage_token_returns_401():
    response = client.get("/me", headers={"Authorization": "Bearer not-a-real-token"})

    assert response.status_code == 401


def test_patch_me_partial_update_email():
    _register()
    token = _login().json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.patch("/me", json={"email": "new@example.com"}, headers=headers)

    assert response.status_code == 200
    assert response.json()["email"] == "new@example.com"

    stale = client.get("/me", headers=headers)
    assert stale.status_code == 401

    relogin = _login(email="new@example.com")
    assert relogin.status_code == 200


def test_patch_me_duplicate_email_returns_400():
    _register(email="first@example.com")
    _register(email="second@example.com")
    token = _login(email="first@example.com").json()["access_token"]

    response = client.patch(
        "/me",
        json={"email": "second@example.com"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


def test_patch_me_password_update_requires_new_password_for_login():
    _register()
    token = _login().json()["access_token"]

    response = client.patch(
        "/me",
        json={"password": "NewStrongPass123"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert _login(password="StrongPass123").status_code == 401
    assert _login(password="NewStrongPass123").status_code == 200
