from __future__ import annotations

from typing import Any

import requests
from streamlit_app.client import ApiClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class RecordingTransport:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.last_post: dict[str, Any] | None = None

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.last_post = {"url": url, **kwargs}
        return self.response


class FailingTransport:
    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        raise requests.ConnectionError("backend unavailable")


def test_login_uses_oauth_form_and_returns_token() -> None:
    transport = RecordingTransport(
        FakeResponse(200, {"access_token": "jwt-token", "token_type": "bearer"})
    )
    client = ApiClient("http://localhost:8000", timeout=5, transport=transport)

    result = client.login("week9@example.com", "strong-password")

    assert result.ok
    assert result.data == {"access_token": "jwt-token", "token_type": "bearer"}
    assert transport.last_post == {
        "url": "http://localhost:8000/api/v1/auth/login",
        "data": {
            "username": "week9@example.com",
            "password": "strong-password",
        },
        "timeout": 5,
    }


def test_predict_sends_bearer_token_and_payload() -> None:
    transport = RecordingTransport(
        FakeResponse(
            201,
            {
                "prediction_history_id": 7,
                "predicted_price": 208_500.0,
                "currency": "USD",
                "model_sha256": "a" * 64,
            },
        )
    )
    client = ApiClient("http://localhost:8000/", timeout=5, transport=transport)
    payload = {"lot_frontage": 70.0, "exter_qual": "Gd"}

    result = client.predict(payload, token="jwt-token")

    assert result.ok
    assert result.data is not None
    assert result.data["predicted_price"] == 208_500.0
    assert transport.last_post == {
        "url": "http://localhost:8000/api/v1/predict",
        "json": payload,
        "headers": {"Authorization": "Bearer jwt-token"},
        "timeout": 5,
    }


def test_connection_error_returns_user_facing_result() -> None:
    client = ApiClient(
        "http://localhost:8000",
        timeout=5,
        transport=FailingTransport(),
    )

    result = client.predict({}, token="jwt-token")

    assert not result.ok
    assert result.data is None
    assert "Cannot connect" in result.message
    assert "http://localhost:8000" in result.message
