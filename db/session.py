"""SQLAlchemy engine and session management."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings


@lru_cache(maxsize=1)
def get_engine(database_url: str | None = None) -> Engine:
    """Create a cached SQLAlchemy engine for the configured database."""

    settings = get_settings()
    return create_engine(
        database_url or settings.database_url,
        echo=settings.sql_echo,
        future=True,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a cached session factory."""

    return sessionmaker(
        bind=get_engine(database_url),
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    """Provide a transactional session scope."""

    session = get_session_factory(database_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_connection_check(database_url: str | None = None) -> None:
    """Execute a lightweight connectivity probe."""

    with get_engine(database_url).connect() as connection:
        connection.execute(text("SELECT 1"))

