"""Shared pytest configuration for the auth flow tests."""

import os

os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-bytes-long")
