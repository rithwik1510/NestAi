"""Application package for the Personal Knowledge Analyst."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking assistance
    from .main import create_app as _create_app

__all__ = ["create_app"]


def create_app():
    """Lazy import to avoid heavy dependencies during module import."""

    from .main import create_app as _create_app

    return _create_app()
