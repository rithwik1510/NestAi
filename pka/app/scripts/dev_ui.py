from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Launch uvicorn with health checks disabled for quick UI previews."""

    os.environ.setdefault("SKIP_HEALTH_CHECKS", "true")
    uvicorn.run(
        "pka.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        factory=False,
    )


if __name__ == "__main__":
    main()
