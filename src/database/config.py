from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# =========================================================
# ENVIRONMENT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# =========================================================
# DATABASE URL
# =========================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. "
        "Create a .env file with the PostgreSQL connection URL."
    )


# =========================================================
# SQLALCHEMY ENGINE
# =========================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# =========================================================
# SESSION FACTORY
# =========================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# =========================================================
# DECLARATIVE BASE
# =========================================================

Base = declarative_base()


def get_db_session():
    """
    Create a new SQLAlchemy database session.

    The caller is responsible for closing the session.
    """

    return SessionLocal()