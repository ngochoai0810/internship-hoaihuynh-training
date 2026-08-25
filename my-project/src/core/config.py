"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings shared by FastAPI and authentication helpers."""

    app_name: str = "House Price API"
    app_version: str = "0.2.0"
    api_v1_prefix: str = "/api/v1"
    secret_key: str = Field(
        default="development-only-change-this-secret",
        min_length=32,
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, gt=0)
    database_url: str = f"sqlite:///{(PROJECT_DIR / 'app.db').as_posix()}"
    model_path: Path = PROJECT_DIR / "model.pkl"
    api_url: str = "http://localhost:8000"
    request_timeout: int = Field(default=10, gt=0)

    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one process-local settings instance."""

    return Settings()
