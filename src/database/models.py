from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database.config import Base


# =========================================================
# HOURLY DEMAND
# =========================================================

class HourlyDemand(Base):
    """
    Processed hourly taxi pickup demand by NYC taxi zone.
    """

    __tablename__ = "hourly_demand"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    pickup_hour_timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    zone_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    demand: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "pickup_hour_timestamp",
            "zone_id",
            name="uq_hourly_demand_timestamp_zone",
        ),
    )


# =========================================================
# PREDICTIONS
# =========================================================

class Prediction(Base):
    """
    Stores model predictions for monitoring and auditing.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    target_timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    zone_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    predicted_demand: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="ExtraTreesRegressor",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )


# =========================================================
# INGESTION METADATA
# =========================================================

class IngestionMetadata(Base):
    """
    Tracks monthly NYC TLC ingestion jobs.

    This allows the ingestion pipeline to determine which
    datasets have already been processed successfully.
    """

    __tablename__ = "ingestion_metadata"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    dataset: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    year: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    month: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    rows_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    processed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "dataset",
            "year",
            "month",
            name="uq_ingestion_dataset_year_month",
        ),
    )