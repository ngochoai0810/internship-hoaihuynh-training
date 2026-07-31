from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routers.houses import get_db
from app.main import app
from orm_migrations.src.models import Base, House


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    db_dir = Path(__file__).resolve().parent / ".test-dbs"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / f"{uuid4().hex}.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    Base.metadata.create_all(engine)

    with testing_session_local() as db:
        db.add_all(
            [
                House(
                    area=120.5,
                    rooms=3,
                    location="District 1",
                    price=250000.0,
                ),
                House(
                    area=85.0,
                    rooms=2,
                    location="District 7",
                    price=180000.0,
                ),
            ]
        )
        db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        try:
            db_dir.rmdir()
        except OSError:
            pass
