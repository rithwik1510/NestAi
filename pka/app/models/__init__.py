"""Database models and schemas."""

from __future__ import annotations

from importlib import import_module
from typing import Dict

__all__ = [
    "Base",
    "Document",
    "Chunk",
    "QARun",
    "QAContext",
    "QAAnswer",
    "HealthProbe",
    "HealthStatus",
]

_MODEL_EXPORTS = {"Base", "Document", "Chunk", "QARun", "QAContext", "QAAnswer"}
_SCHEMA_EXPORTS = {"HealthProbe", "HealthStatus"}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = import_module(".db", __name__)
        return getattr(module, name)
    if name in _SCHEMA_EXPORTS:
        module = import_module(".schema", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
