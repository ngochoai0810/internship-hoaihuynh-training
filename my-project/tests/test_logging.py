from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from api.main import create_app
from core.config import Settings
from fastapi.testclient import TestClient
from models import Base


class LoggingModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([np.log1p(200_000.0)])


def _create_logging_app(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    joblib.dump(LoggingModel(), model_path)
    app = create_app(
        Settings(
            secret_key="test-secret-key-that-is-long-enough",
            database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
            model_path=model_path,
            log_level="INFO",
        )
    )
    Base.metadata.create_all(app.state.engine)
    return app


def test_settings_exposes_log_level_override() -> None:
    settings = Settings(
        secret_key="test-secret-key-that-is-long-enough",
        log_level="DEBUG",
    )

    assert settings.log_level == "DEBUG"


def test_request_logging_includes_metadata_and_hides_sensitive_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _create_logging_app(tmp_path)

    with caplog.at_level(logging.INFO, logger="api.request"):
        with TestClient(app) as client:
            response = client.get(
                "/health",
                headers={"Authorization": "Bearer secret-token-value"},
            )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    log_text = caplog.text
    assert "method=GET" in log_text
    assert "path=/health" in log_text
    assert "status_code=200" in log_text
    assert "duration_ms=" in log_text
    assert "request_id=" in log_text
    assert "secret-token-value" not in log_text
    assert "Authorization" not in log_text
