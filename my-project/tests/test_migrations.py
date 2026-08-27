from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_upgrade_head_creates_week9_tables(tmp_path: Path) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "migration.db"
    config = Config(project_dir / "alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        inspector = inspect(engine)
        assert {"users", "prediction_history"} <= set(inspector.get_table_names())
        user_indexes = inspector.get_indexes("users")
        assert any(
            index["unique"] and index["column_names"] == ["email"]
            for index in user_indexes
        )
    finally:
        engine.dispose()


def test_upgrade_head_uses_environment_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "environment.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    config = Config(project_dir / "alembic.ini")
    command.upgrade(config, "head")

    engine = create_engine(database_url)
    try:
        assert {"users", "prediction_history"} <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
