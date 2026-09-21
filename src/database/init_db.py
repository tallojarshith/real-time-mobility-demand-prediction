from __future__ import annotations

from sqlalchemy import text

from src.database.config import Base, engine
from src.database.models import (
    HourlyDemand,
    IngestionMetadata,
    Prediction,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def test_database_connection() -> None:
    """
    Verify that PostgreSQL is reachable.
    """

    logger.info(
        "Testing PostgreSQL connection"
    )

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        value = result.scalar()

        if value != 1:
            raise RuntimeError(
                "Unexpected PostgreSQL health-check result"
            )

    logger.info(
        "PostgreSQL connection successful"
    )


def create_database_tables() -> None:
    """
    Create all SQLAlchemy tables that do not already exist.

    create_all() is idempotent for existing tables.
    """

    logger.info(
        "Creating database tables"
    )

    Base.metadata.create_all(
        bind=engine
    )

    logger.info(
        "Database tables created successfully"
    )


def initialize_database() -> None:
    """
    Test PostgreSQL and initialize application tables.
    """

    test_database_connection()

    create_database_tables()


if __name__ == "__main__":

    initialize_database()

    print()
    print("=" * 50)
    print("DATABASE INITIALIZATION PASSED")
    print("=" * 50)

    print(
        "PostgreSQL connection: OK"
    )

    print(
        "Tables:"
    )

    print(
        "  - hourly_demand"
    )

    print(
        "  - predictions"
    )

    print(
        "  - ingestion_metadata"
    )