from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select

from src.database.config import SessionLocal
from src.database.models import HourlyDemand
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_zone_demand_history(
    zone_id: int,
    target_timestamp: datetime,
    history_hours: int = 168,
) -> pd.DataFrame:
    """
    Retrieve historical hourly demand for one taxi zone.

    For a target timestamp T, the returned history covers:

        T - history_hours  through  T - 1 hour

    The target hour itself is never included, preventing
    target leakage during inference.
    """

    if zone_id < 1 or zone_id > 265:
        raise ValueError(
            f"Invalid taxi zone: {zone_id}"
        )

    if history_hours <= 0:
        raise ValueError(
            "history_hours must be greater than zero"
        )

    # Normalize target timestamp to an exact hour.
    target_hour = target_timestamp.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    history_start = (
        target_hour
        - timedelta(hours=history_hours)
    )

    logger.info(
        "Reading demand history from PostgreSQL | "
        "zone=%d | start=%s | end=%s",
        zone_id,
        history_start,
        target_hour,
    )

    session = SessionLocal()

    try:

        statement = (
            select(
                HourlyDemand.pickup_hour_timestamp,
                HourlyDemand.zone_id,
                HourlyDemand.demand,
            )
            .where(
                HourlyDemand.zone_id == zone_id,
                HourlyDemand.pickup_hour_timestamp
                >= history_start,
                HourlyDemand.pickup_hour_timestamp
                < target_hour,
            )
            .order_by(
                HourlyDemand.pickup_hour_timestamp
            )
        )

        rows = session.execute(
            statement
        ).all()

    finally:

        session.close()

    if not rows:
        raise ValueError(
            "No demand history found in PostgreSQL for "
            f"zone {zone_id} before {target_hour}."
        )

    history = pd.DataFrame(
        rows,
        columns=[
            "pickup_hour_timestamp",
            "PULocationID",
            "demand",
        ],
    )

    history["pickup_hour_timestamp"] = (
        pd.to_datetime(
            history["pickup_hour_timestamp"]
        )
    )

    history["PULocationID"] = (
        history["PULocationID"].astype(int)
    )

    history["demand"] = (
        history["demand"].astype(float)
    )

    # -----------------------------------------------------
    # Validate complete history
    # -----------------------------------------------------

    expected_timestamps = pd.date_range(
        start=history_start,
        end=target_hour,
        freq="h",
        inclusive="left",
    )

    actual_timestamps = pd.DatetimeIndex(
        history["pickup_hour_timestamp"]
    )

    missing_timestamps = (
        expected_timestamps.difference(
            actual_timestamps
        )
    )

    if len(missing_timestamps) > 0:

        raise ValueError(
            "Demand history is incomplete for "
            f"zone {zone_id}. "
            f"Expected {history_hours} hours but "
            f"found {len(history)}. "
            f"Missing {len(missing_timestamps)} hourly timestamps."
        )

    if len(history) != history_hours:

        raise ValueError(
            f"Expected exactly {history_hours} demand rows "
            f"for zone {zone_id}, but found {len(history)}."
        )

    logger.info(
        "Demand history retrieved successfully | "
        "zone=%d | rows=%d",
        zone_id,
        len(history),
    )

    return history