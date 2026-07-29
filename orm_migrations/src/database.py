"""SQLAlchemy database connection and session setup."""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

ORM_MIGRATIONS_ROOT = Path(__file__).resolve().parents[1]
SQLALCHEMY_DATABASE_URL = f"sqlite:///{(ORM_MIGRATIONS_ROOT / 'app.db').as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Open a database session and close it after use."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
