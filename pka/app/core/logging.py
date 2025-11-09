from __future__ import annotations

import logging
import sys
from typing import Optional

from .settings import settings


def configure_logging(level: Optional[str] = None) -> None:
    """Configure structured logging for the application."""

    log_level = (level or settings.log_level).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


__all__ = ["configure_logging"]
