from __future__ import annotations

from datetime import datetime

from src.database.config import SessionLocal
from src.database.models import Prediction
from src.utils.logger import get_logger


logger = get_logger(__name__)


def save_prediction(
    target_timestamp: datetime,
    zone_id: int,
    predicted_demand: float,
    model_name: str = "ExtraTreesRegressor",
) -> int:
    """
    Save a generated taxi-demand prediction to PostgreSQL.

    Returns the database ID of the stored prediction.
    """

    session = SessionLocal()

    try:
        prediction_record = Prediction(
            target_timestamp=target_timestamp,
            zone_id=zone_id,
            predicted_demand=float(predicted_demand),
            model_name=model_name,
        )

        session.add(prediction_record)
        session.commit()
        session.refresh(prediction_record)

        logger.info(
            "Prediction stored in PostgreSQL | "
            "id=%d | zone=%d | timestamp=%s | prediction=%.2f",
            prediction_record.id,
            zone_id,
            target_timestamp,
            predicted_demand,
        )

        return prediction_record.id

    except Exception:
        session.rollback()

        logger.exception(
            "Failed to store prediction in PostgreSQL"
        )

        raise

    finally:
        session.close()