"""Create the local SQLite database schema."""

from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from orm_migrations.src.database import engine
from orm_migrations.src.models import Base


def init_db() -> None:
    """Create all registered ORM tables in the local SQLite database."""

    Base.metadata.create_all(bind=engine)
    table_names = ", ".join(sorted(Base.metadata.tables))
    print(f"Created SQLite schema for {table_names} in app.db")


if __name__ == "__main__":
    init_db()
