"""
SQLAlchemy 2.0 core setup.

Includes:
- `engine`: connection setup for local SQLite (app.db)
- `SessionLocal`: factory that creates Session objects
- `Base`: declarative base class for ORM models
- `get_db`: FastAPI dependency with yield
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in this project."""

    pass


def get_db() -> Generator[Session, None, None]:
    """Open a database session and close it after use."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
