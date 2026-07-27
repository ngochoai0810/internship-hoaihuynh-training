from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from orm_migrations.src.models.base import Base
from orm_migrations.src.models.house import House


def _make_test_session(
    database_url: str = "sqlite:///:memory:",
) -> tuple[Session, object]:
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine)
    return testing_session_local(), engine


def test_house_table_created_with_expected_columns() -> None:
    _, engine = _make_test_session()

    inspector = inspect(engine)
    assert "houses" in inspector.get_table_names()

    columns = {col["name"] for col in inspector.get_columns("houses")}
    expected = {
        "id",
        "area",
        "rooms",
        "location",
        "price",
        "created_at",
    }
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


def test_house_table_created_in_sqlite_file() -> None:
    cache_dir = Path(".pytest_cache") / "sqlite-tests"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / f"{uuid4().hex}.db"

    _, engine = _make_test_session(f"sqlite:///{db_path}")

    assert db_path.exists()

    inspector = inspect(engine)
    assert "houses" in inspector.get_table_names()
