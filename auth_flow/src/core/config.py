"""Application security configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def required_env(name: str) -> str:
    """Read a required environment variable or fail during startup."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be set in the environment or .env file")
    return value


class Settings:
    """Central place for auth-related configuration values."""

    SECRET_KEY: str = required_env("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )


settings = Settings()
