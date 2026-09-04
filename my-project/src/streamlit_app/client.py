"""HTTP boundary used by the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


class HttpTransport(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...

    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    message: str
    data: dict[str, Any] | None = None


class ApiClient:
    """Small synchronous client for the Week 9 FastAPI contract."""

    def __init__(
        self,
        api_url: str,
        *,
        timeout: int,
        transport: HttpTransport | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or requests

    def _error(self, response: Any) -> ApiResult:
        try:
            detail = response.json().get("detail", "Request failed")
        except (AttributeError, ValueError):
            detail = "Request failed"
        if isinstance(detail, list):
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        return ApiResult(False, str(detail))

    def health(self) -> ApiResult:
        try:
            response = self.transport.get(
                f"{self.api_url}/health",
                timeout=self.timeout,
            )
        except requests.RequestException:
            return ApiResult(False, f"Cannot connect to {self.api_url}")
        if response.status_code == 200:
            return ApiResult(True, "Backend is ready", response.json())
        return self._error(response)

    def register(self, email: str, password: str) -> ApiResult:
        try:
            response = self.transport.post(
                f"{self.api_url}/api/v1/auth/register",
                json={"email": email, "password": password},
                timeout=self.timeout,
            )
        except requests.RequestException:
            return ApiResult(False, f"Cannot connect to {self.api_url}")
        if response.status_code == 201:
            return ApiResult(True, "Registration successful", response.json())
        return self._error(response)

    def login(self, email: str, password: str) -> ApiResult:
        try:
            response = self.transport.post(
                f"{self.api_url}/api/v1/auth/login",
                data={"username": email, "password": password},
                timeout=self.timeout,
            )
        except requests.RequestException:
            return ApiResult(False, f"Cannot connect to {self.api_url}")
        if response.status_code == 200:
            return ApiResult(True, "Login successful", response.json())
        return self._error(response)

    def predict(self, payload: dict[str, Any], *, token: str) -> ApiResult:
        try:
            response = self.transport.post(
                f"{self.api_url}/api/v1/predict",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
        except requests.RequestException:
            return ApiResult(False, f"Cannot connect to {self.api_url}")
        if response.status_code == 201:
            return ApiResult(True, "Prediction saved", response.json())
        if response.status_code == 401:
            return ApiResult(False, "Session expired. Please log in again.")
        return self._error(response)

    def prediction_history(self, *, token: str, limit: int = 20) -> ApiResult:
        try:
            response = self.transport.get(
                f"{self.api_url}/api/v1/predictions/history",
                headers={"Authorization": f"Bearer {token}"},
                params={"limit": limit},
                timeout=self.timeout,
            )
        except requests.RequestException:
            return ApiResult(False, f"Cannot connect to {self.api_url}")
        if response.status_code == 200:
            return ApiResult(
                True, "Prediction history loaded", {"items": response.json()}
            )
        if response.status_code == 401:
            return ApiResult(False, "Session expired. Please log in again.")
        return self._error(response)
