"""Shared logging configuration for API and operational scripts."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure process logging with a compact stdout-friendly format."""

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )
