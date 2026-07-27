"""Create the local SQLite database schema."""

from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from orm_migrations.src.models.base import Base, engine
from orm_migrations.src.models.house import House


def init_db() -> None:
    """Create all registered ORM tables in the local SQLite database."""

    Base.metadata.create_all(bind=engine)
    print(f"Created SQLite schema for {House.__tablename__!r} in app.db")


if __name__ == "__main__":
    init_db()
