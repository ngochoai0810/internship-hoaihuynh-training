"""
Application security configuration.

Loads sensitive values from a `.env` file instead of hardcoding secrets in
source. Never commit the real `.env` file.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central place for auth-related configuration values."""

    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "change-me-in-dot-env-at-least-32-bytes"
    )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )


settings = Settings()
