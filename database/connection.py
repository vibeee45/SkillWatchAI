"""Database connection and session management for SkillWatch AI.

Supports local SQLite by default for edge zero-config operation,
and PostgreSQL when DATABASE_URL or SKILLWATCH_DATABASE_URL is provided.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SQLITE_URL = f"sqlite:///{DATA_DIR / 'skillwatch.db'}"
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("SKILLWATCH_DATABASE_URL") or DEFAULT_SQLITE_URL

# SQLite requires check_same_thread=False for multi-threaded FastAPI workers
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    # Import models to ensure they are registered with Base.metadata
    from database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
