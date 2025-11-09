"""Document-related services."""

from importlib import import_module

__all__ = ["DocumentService", "DocumentView", "DocumentChunkView"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".service", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
