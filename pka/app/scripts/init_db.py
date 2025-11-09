from __future__ import annotations

from ..core.logging import configure_logging
from ..models.db import Base, engine


def main() -> None:
    """Create database tables using the SQLAlchemy metadata."""

    configure_logging()
    Base.metadata.create_all(bind=engine)
    print("Database schema ensured (tables created if missing).")


if __name__ == "__main__":
    main()
