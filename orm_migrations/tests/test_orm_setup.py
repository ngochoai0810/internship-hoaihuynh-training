from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from orm_migrations.src.models import Base, House, PredictionHistory, User


def _make_test_session(
    database_url: str = "sqlite:///:memory:",
) -> tuple[Session, object]:
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine)
    return testing_session_local(), engine


def test_tables_created_with_expected_columns() -> None:
    _, engine = _make_test_session()

    inspector = inspect(engine)
    assert {"houses", "users", "prediction_history"}.issubset(
        set(inspector.get_table_names())
    )

    expected_columns = {
        "houses": {"id", "created_at", "updated_at", "area", "rooms", "location", "price"},
        "users": {"id", "created_at", "updated_at", "email", "hashed_password"},
        "prediction_history": {
            "id",
            "created_at",
            "updated_at",
            "user_id",
            "input_area",
            "input_rooms",
            "input_location",
            "predicted_price",
        },
    }

    for table_name, expected in expected_columns.items():
        columns = {col["name"] for col in inspector.get_columns(table_name)}
        assert expected.issubset(columns)


def test_house_insert_and_query() -> None:
    db, _ = _make_test_session()

    house = House(
        area=120.5,
        rooms=3,
        location="District 1",
        price=250000.0,
    )
    db.add(house)
    db.commit()
    db.refresh(house)

    assert house.id is not None

    fetched = db.query(House).filter(House.location == "District 1").first()
    assert fetched is not None
    assert fetched.area == 120.5

    db.close()


def test_user_prediction_relationship_works_both_ways() -> None:
    db, _ = _make_test_session()

    user = User(email="hoai@example.com", hashed_password="not-plain-text")
    prediction = PredictionHistory(
        user=user,
        input_area=120.5,
        input_rooms=3,
        input_location="District 1",
        predicted_price=250000.0,
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    db.refresh(prediction)

    assert prediction.user_id == user.id
    assert user.predictions == [prediction]
    assert prediction.user.email == "hoai@example.com"

    db.close()


def test_user_delete_cascades_to_prediction_history() -> None:
    db, _ = _make_test_session()

    user = User(email="cascade@example.com", hashed_password="not-plain-text")
    prediction = PredictionHistory(
        user=user,
        input_area=95.0,
        input_rooms=2,
        input_location="District 3",
        predicted_price=180000.0,
    )

    db.add(user)
    db.commit()
    prediction_id = prediction.id

    db.delete(user)
    db.commit()

    deleted_prediction = db.scalar(
        select(PredictionHistory).where(PredictionHistory.id == prediction_id)
    )
    assert deleted_prediction is None

    db.close()


def test_house_table_created_in_sqlite_file() -> None:
    db_path = Path(__file__).resolve().parent / f"{uuid4().hex}.db"
    engine = None

    try:
        _, engine = _make_test_session(f"sqlite:///{db_path}")

        assert db_path.exists()

        inspector = inspect(engine)
        assert "houses" in inspector.get_table_names()
    finally:
        if engine is not None:
            engine.dispose()
        db_path.unlink(missing_ok=True)
