from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def optional_api_key(key: str | None = Depends(API_KEY_HEADER)) -> str | None:
    """Placeholder for future API key validation."""

    if not key:
        return None
    # Real validation to be implemented once authentication requirements are defined.
    return key


async def require_api_key(key: str | None = Depends(optional_api_key)) -> str:
    if key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required.")
    return key


__all__ = ["optional_api_key", "require_api_key"]
