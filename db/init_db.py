"""Database bootstrap helpers."""

from __future__ import annotations

from db.schema import Base
from db.session import get_engine


def create_all_tables(database_url: str | None = None) -> None:
    """Create all declarative tables in the configured database."""

    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_all_tables()

