from __future__ import annotations

import hashlib
from collections.abc import Generator
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from api.main import create_app
from core.config import Settings
from core.security import create_access_token
from database import get_db
from fastapi.testclient import TestClient
from models import Base, PredictionHistory, User
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

VALID_PREDICTION = {
    "lot_frontage": 70.0,
    "mas_vnr_area": 100.0,
    "total_bsmt_sf": 856.0,
    "garage_type": "Attchd",
    "alley": None,
    "exter_qual": "Gd",
}


class InspectingModel:
    """Small serializable model double that enforces the inference contract."""

    def __init__(self, price: float) -> None:
        self.price = price

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert list(frame.columns) == [
            "LotFrontage",
            "MasVnrArea",
            "TotalBsmtSF",
            "GarageType",
            "Alley",
            "ExterQual",
        ]
        return np.array([np.log1p(self.price)])


class FailingModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise ValueError("inference failed")


class OverflowingModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([1_000.0])


class NegativePriceModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([-1.0])


def _settings(tmp_path: Path, model_path: Path) -> Settings:
    return Settings(
        secret_key="test-secret-key-that-is-long-enough",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        model_path=model_path,
    )


def _create_test_app(tmp_path: Path, model: object):
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    app = create_app(_settings(tmp_path, model_path))
    Base.metadata.create_all(app.state.engine)
    return app, model_path


def _register_and_login(client: TestClient) -> str:
    register_response = client.post(
        "/api/v1/auth/register",
        json={"email": "week9@example.com", "password": "strong-password"},
    )
    assert register_response.status_code == 201
    assert set(register_response.json()) == {"id", "email"}

    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "week9@example.com", "password": "strong-password"},
    )
    assert login_response.status_code == 200
    return str(login_response.json()["access_token"])


def test_lifespan_loads_artifact_and_health_reports_fingerprint(
    tmp_path: Path,
) -> None:
    app, model_path = _create_test_app(tmp_path, InspectingModel(208_500.0))
    expected_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_sha256": expected_sha256,
    }


@pytest.mark.parametrize("artifact", [None, b"not-a-joblib-model"])
def test_lifespan_rejects_missing_or_corrupt_model(
    tmp_path: Path,
    artifact: bytes | None,
) -> None:
    model_path = tmp_path / "model.pkl"
    if artifact is not None:
        model_path.write_bytes(artifact)
    app = create_app(_settings(tmp_path, model_path))

    with pytest.raises(RuntimeError, match="model artifact"):
        with TestClient(app):
            pass


def test_authenticated_prediction_is_persisted(tmp_path: Path) -> None:
    app, model_path = _create_test_app(tmp_path, InspectingModel(208_500.0))
    expected_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    with TestClient(app) as client:
        token = _register_and_login(client)
        response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 201
    assert response.json() == {
        "prediction_history_id": 1,
        "predicted_price": 208_500.0,
        "currency": "USD",
        "model_sha256": expected_sha256,
    }

    with app.state.session_factory() as session:
        history = session.scalar(select(PredictionHistory))
        assert history is not None
        assert history.input_payload == VALID_PREDICTION
        assert history.predicted_price == 208_500.0
        assert history.model_sha256 == expected_sha256
        assert history.user.email == "week9@example.com"


def test_missing_token_does_not_create_history(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, InspectingModel(208_500.0))

    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=VALID_PREDICTION)

    assert response.status_code == 401
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_inference_error_does_not_create_history(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, FailingModel())

    with TestClient(app, raise_server_exceptions=False) as client:
        token = _register_and_login(client)
        response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_inverse_transform_overflow_does_not_create_history(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, OverflowingModel())

    with TestClient(app, raise_server_exceptions=False) as client:
        token = _register_and_login(client)
        response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_negative_price_does_not_create_history(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, NegativePriceModel())

    with TestClient(app, raise_server_exceptions=False) as client:
        token = _register_and_login(client)
        response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_database_error_rolls_back_prediction_history(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, InspectingModel(208_500.0))
    with app.state.session_factory() as session:
        user = User(email="database-error@example.com", hashed_password="unused")
        session.add(user)
        session.commit()
        user_id = user.id

    class CommitFailingSession(Session):
        def commit(self) -> None:
            raise SQLAlchemyError("database write failed")

    def failing_db() -> Generator[Session, None, None]:
        with CommitFailingSession(bind=app.state.engine) as session:
            yield session

    app.dependency_overrides[get_db] = failing_db
    token = create_access_token(str(user_id), settings=app.state.settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_restart_loads_overwritten_model_without_route_changes(
    tmp_path: Path,
) -> None:
    app, model_path = _create_test_app(tmp_path, InspectingModel(100_000.0))

    with TestClient(app) as client:
        token = _register_and_login(client)
        first_health = client.get("/health").json()
        first_prediction = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        ).json()

    joblib.dump(InspectingModel(300_000.0), model_path)

    with TestClient(app) as restarted_client:
        second_health = restarted_client.get("/health").json()
        second_prediction = restarted_client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {token}"},
        ).json()

    assert first_health["model_sha256"] != second_health["model_sha256"]
    assert first_prediction["predicted_price"] == 100_000.0
    assert second_prediction["predicted_price"] == 300_000.0


def test_invalid_and_expired_tokens_are_rejected(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, InspectingModel(208_500.0))

    with TestClient(app) as client:
        _register_and_login(client)
        expired_token = create_access_token(
            "1",
            settings=app.state.settings,
            expires_delta=timedelta(seconds=-1),
        )
        invalid_response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": "Bearer invalid-token"},
        )
        expired_response = client.post(
            "/api/v1/predict",
            json=VALID_PREDICTION,
            headers={"Authorization": f"Bearer {expired_token}"},
        )

    assert invalid_response.status_code == 401
    assert expired_response.status_code == 401
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_invalid_prediction_payload_is_rejected(tmp_path: Path) -> None:
    app, _ = _create_test_app(tmp_path, InspectingModel(208_500.0))

    with TestClient(app) as client:
        token = _register_and_login(client)
        response = client.post(
            "/api/v1/predict",
            json={**VALID_PREDICTION, "lot_frontage": -1},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(PredictionHistory.id))) == 0


def test_registration_rejects_password_beyond_bcrypt_byte_limit(
    tmp_path: Path,
) -> None:
    app, _ = _create_test_app(tmp_path, InspectingModel(208_500.0))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "long-password@example.com", "password": "x" * 73},
        )

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count(User.id))) == 0
