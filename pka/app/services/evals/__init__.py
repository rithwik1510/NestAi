"""Evaluation utilities and datasets."""

from __future__ import annotations

__all__ = ["EvaluationRunner"]


def __getattr__(name: str):
    if name == "EvaluationRunner":
        from .scorer import EvaluationRunner  # local import for lightweight module loading

        return EvaluationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
