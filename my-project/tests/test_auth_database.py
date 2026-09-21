from datetime import timedelta

from core.config import Settings
from core.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from models import Base, PredictionHistory, User
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session


def test_user_and_prediction_history_persist_expected_data() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(email="week9@example.com", hashed_password="hashed")
        history = PredictionHistory(
            user=user,
            input_payload={
                "overall_qual": 7,
                "garage_type": "Attchd",
            },
            predicted_price=208_500.0,
            model_sha256="a" * 64,
        )
        session.add(user)
        session.commit()
        session.refresh(history)

        assert history.user_id == user.id
        assert history.input_payload["overall_qual"] == 7
        assert user.predictions == [history]

    columns = {
        column["name"] for column in inspect(engine).get_columns("prediction_history")
    }
    assert {
        "id",
        "user_id",
        "input_payload",
        "predicted_price",
        "model_sha256",
        "created_at",
        "updated_at",
    } <= columns
    engine.dispose()


def test_password_hash_and_jwt_round_trip() -> None:
    settings = Settings(
        secret_key="test-secret-key-that-is-long-enough",
        access_token_expire_minutes=30,
    )

    hashed_password = hash_password("correct horse battery staple")
    token = create_access_token("42", settings=settings)

    assert hashed_password != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed_password)
    assert not verify_password("wrong password", hashed_password)
    assert decode_access_token(token, settings=settings) == "42"


def test_expired_jwt_is_rejected() -> None:
    settings = Settings(secret_key="test-secret-key-that-is-long-enough")
    token = create_access_token(
        "42",
        settings=settings,
        expires_delta=timedelta(seconds=-1),
    )

    try:
        decode_access_token(token, settings=settings)
    except InvalidTokenError:
        pass
    else:
        raise AssertionError("Expired JWT must be rejected")
