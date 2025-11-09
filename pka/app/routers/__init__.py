"""API routers exposed by the application."""

from importlib import import_module

__all__ = ["health", "chat", "web"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
