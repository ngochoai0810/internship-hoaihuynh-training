"""Seed sample users and prediction history rows idempotently."""

from pathlib import Path
import os
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

ORM_MIGRATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ORM_MIGRATIONS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(ORM_MIGRATIONS_ROOT)

from orm_migrations.src.database import SessionLocal  # noqa: E402
from orm_migrations.src.models import PredictionHistory, User  # noqa: E402


SEED_USERS = [
    {
        "email": "alice@example.com",
        "hashed_password": "not-plain-text-alice",
        "predictions": [
            ("District 1", 120.5, 3, 250000.0),
            ("District 3", 88.0, 2, 175000.0),
            ("Thu Duc", 64.0, 2, 120000.0),
        ],
    },
    {
        "email": "bob@example.com",
        "hashed_password": "not-plain-text-bob",
        "predictions": [
            ("District 7", 95.0, 2, 210000.0),
            ("Binh Thanh", 72.5, 2, 160000.0),
        ],
    },
]


def get_or_create_user(session: Session, email: str, hashed_password: str) -> User:
    """Return an existing user by email, or insert one when it is missing."""

    stmt = select(User).where(User.email == email)
    user = session.execute(stmt).scalar_one_or_none()
    if user is not None:
        print(f"[skip] User exists: {user}")
        return user

    user = User(email=email, hashed_password=hashed_password)
    session.add(user)
    session.flush()
    print(f"[created] {user}")
    return user


def get_or_create_prediction(
    session: Session,
    user: User,
    input_location: str,
    input_area: float,
    input_rooms: int,
    predicted_price: float,
) -> PredictionHistory:
    """Return an existing prediction for the same user/input, or insert it."""

    stmt = select(PredictionHistory).where(
        PredictionHistory.user_id == user.id,
        PredictionHistory.input_location == input_location,
        PredictionHistory.input_area == input_area,
        PredictionHistory.input_rooms == input_rooms,
    )
    prediction = session.execute(stmt).scalar_one_or_none()
    if prediction is not None:
        print(f"[skip] Prediction exists: {prediction}")
        return prediction

    prediction = PredictionHistory(
        user=user,
        input_location=input_location,
        input_area=input_area,
        input_rooms=input_rooms,
        predicted_price=predicted_price,
    )
    session.add(prediction)
    session.flush()
    print(f"[created] {prediction}")
    return prediction


def run_seed() -> None:
    """Insert all sample rows, skipping rows already present."""

    session = SessionLocal()
    try:
        for seed_user in SEED_USERS:
            user = get_or_create_user(
                session,
                email=seed_user["email"],
                hashed_password=seed_user["hashed_password"],
            )
            for location, area, rooms, price in seed_user["predictions"]:
                get_or_create_prediction(session, user, location, area, rooms, price)

        session.commit()
        print("Seed completed.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    run_seed()
