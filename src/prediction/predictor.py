import numpy as np
import pandas as pd

from src.models.model_loader import load_model_and_features
from src.prediction.inference_features import build_inference_features
from src.utils.logger import get_logger


# =========================================================
# LOGGER
# =========================================================

logger = get_logger(__name__)


# =========================================================
# DEMAND PREDICTOR
# =========================================================

class DemandPredictor:
    """
    Production prediction service for NYC taxi demand.

    Supports:
    1. Single-zone next-hour prediction.
    2. Multi-zone next-hour batch prediction.

    The trained model is loaded only once when the
    DemandPredictor object is created.
    """

    def __init__(self):
        """
        Load the trained model and exact feature schema.
        """

        logger.info(
            "Initializing taxi demand predictor"
        )

        self.model, self.feature_columns = (
            load_model_and_features()
        )

        logger.info(
            "Taxi demand predictor initialized successfully "
            "with %d features",
            len(self.feature_columns)
        )

    # =====================================================
    # SINGLE-ZONE PREDICTION
    # =====================================================

    def predict(
        self,
        demand_history: pd.DataFrame,
        weather_row: pd.DataFrame,
        target_timestamp,
        zone_id: int
    ) -> float:
        """
        Predict next-hour taxi pickup demand for one zone.

        Parameters
        ----------
        demand_history : pd.DataFrame
            Historical zone-hour demand containing:
                pickup_hour_timestamp
                PULocationID
                demand

        weather_row : pd.DataFrame
            Weather information for the target hour.

        target_timestamp :
            Hour for which demand should be predicted.

        zone_id : int
            NYC taxi-zone ID.

        Returns
        -------
        float
            Predicted pickup demand.
        """

        target_timestamp = pd.Timestamp(
            target_timestamp
        )

        zone_id = int(
            zone_id
        )

        logger.info(
            "Generating single-zone prediction | "
            "zone=%d | timestamp=%s",
            zone_id,
            target_timestamp
        )

        # -------------------------------------------------
        # 1. Build leakage-safe inference features
        # -------------------------------------------------

        features = build_inference_features(
            demand_history=demand_history,
            weather_row=weather_row,
            target_timestamp=target_timestamp,
            zone_id=zone_id
        )

        # -------------------------------------------------
        # 2. Validate expected model features
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in features.columns
        ]

        if missing_features:

            logger.error(
                "Missing model features: %s",
                missing_features
            )

            raise ValueError(
                f"Missing model features: "
                f"{missing_features}"
            )

        # -------------------------------------------------
        # 3. Enforce exact feature order
        # -------------------------------------------------

        model_input = features[
            self.feature_columns
        ]

        # -------------------------------------------------
        # 4. Generate prediction
        # -------------------------------------------------

        prediction = self.model.predict(
            model_input
        )[0]

        prediction = float(
            prediction
        )

        # -------------------------------------------------
        # 5. Validate prediction
        # -------------------------------------------------

        if not np.isfinite(prediction):

            logger.error(
                "Model produced non-finite prediction | "
                "zone=%d | prediction=%s",
                zone_id,
                prediction
            )

            raise ValueError(
                "Model produced a non-finite prediction"
            )

        # Demand cannot be negative.
        prediction = max(
            0.0,
            prediction
        )

        logger.info(
            "Prediction generated | "
            "zone=%d | timestamp=%s | "
            "predicted_demand=%.2f",
            zone_id,
            target_timestamp,
            prediction
        )

        return prediction

    # =====================================================
    # MULTI-ZONE PREDICTION
    # =====================================================

    def predict_multiple_zones(
        self,
        demand_history: pd.DataFrame,
        weather_row: pd.DataFrame,
        target_timestamp,
        zone_ids=None
    ) -> pd.DataFrame:
        """
        Predict next-hour demand for multiple taxi zones.

        If zone_ids is None, all zones available in
        demand_history are used.

        Zones without sufficient historical information
        are skipped and logged.

        Returns
        -------
        pd.DataFrame
            Columns:
                pickup_hour_timestamp
                PULocationID
                predicted_demand
        """

        target_timestamp = pd.Timestamp(
            target_timestamp
        )

        # -------------------------------------------------
        # 1. Validate demand history
        # -------------------------------------------------

        required_columns = [
            "pickup_hour_timestamp",
            "PULocationID",
            "demand"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in demand_history.columns
        ]

        if missing_columns:

            logger.error(
                "Demand history missing columns: %s",
                missing_columns
            )

            raise ValueError(
                f"Missing demand history columns: "
                f"{missing_columns}"
            )

        # -------------------------------------------------
        # 2. Determine zones to predict
        # -------------------------------------------------

        if zone_ids is None:

            zone_ids = sorted(
                demand_history[
                    "PULocationID"
                ]
                .dropna()
                .unique()
            )

        zone_ids = [
            int(zone_id)
            for zone_id in zone_ids
        ]

        if len(zone_ids) == 0:

            raise ValueError(
                "No taxi zones supplied for prediction"
            )

        logger.info(
            "Starting multi-zone prediction | "
            "zones=%d | timestamp=%s",
            len(zone_ids),
            target_timestamp
        )

        # -------------------------------------------------
        # 3. Build inference features for each zone
        # -------------------------------------------------

        feature_frames = []

        valid_zone_ids = []

        skipped_zones = []

        for zone_id in zone_ids:

            try:

                zone_features = (
                    build_inference_features(
                        demand_history=demand_history,
                        weather_row=weather_row,
                        target_timestamp=target_timestamp,
                        zone_id=zone_id
                    )
                )

                feature_frames.append(
                    zone_features
                )

                valid_zone_ids.append(
                    zone_id
                )

            except ValueError as error:

                skipped_zones.append(
                    zone_id
                )

                logger.warning(
                    "Skipping zone=%d | reason=%s",
                    zone_id,
                    error
                )

        # -------------------------------------------------
        # 4. Ensure at least one zone is usable
        # -------------------------------------------------

        if not feature_frames:

            logger.error(
                "No zones had sufficient historical "
                "information for prediction"
            )

            raise ValueError(
                "No zones had sufficient history "
                "for prediction"
            )

        # -------------------------------------------------
        # 5. Combine all zone features
        # -------------------------------------------------

        model_input = pd.concat(
            feature_frames,
            ignore_index=True
        )

        # -------------------------------------------------
        # 6. Validate feature schema
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in model_input.columns
        ]

        if missing_features:

            logger.error(
                "Missing model features during "
                "multi-zone prediction: %s",
                missing_features
            )

            raise ValueError(
                f"Missing model features: "
                f"{missing_features}"
            )

        # Exact trained feature order
        model_input = model_input[
            self.feature_columns
        ]

        # -------------------------------------------------
        # 7. Batch prediction
        # -------------------------------------------------

        predictions = self.model.predict(
            model_input
        )

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        # -------------------------------------------------
        # 8. Validate predictions
        # -------------------------------------------------

        if not np.isfinite(
            predictions
        ).all():

            logger.error(
                "Model produced non-finite values "
                "during multi-zone prediction"
            )

            raise ValueError(
                "Model produced non-finite predictions"
            )

        # Demand cannot be negative
        predictions = np.maximum(
            predictions,
            0.0
        )

        # -------------------------------------------------
        # 9. Create result dataframe
        # -------------------------------------------------

        results = pd.DataFrame({
            "pickup_hour_timestamp":
                [target_timestamp]
                * len(valid_zone_ids),

            "PULocationID":
                valid_zone_ids,

            "predicted_demand":
                predictions
        })

        # Highest-demand zones first
        results = results.sort_values(
            by="predicted_demand",
            ascending=False
        ).reset_index(
            drop=True
        )

        # -------------------------------------------------
        # 10. Logging
        # -------------------------------------------------

        logger.info(
            "Multi-zone prediction completed | "
            "requested=%d | predicted=%d | skipped=%d",
            len(zone_ids),
            len(results),
            len(skipped_zones)
        )

        if skipped_zones:

            logger.warning(
                "Skipped zones: %s",
                skipped_zones
            )

        return results


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def predict_demand(
    demand_history: pd.DataFrame,
    weather_row: pd.DataFrame,
    target_timestamp,
    zone_id: int
) -> float:
    """
    Convenience function for generating one prediction.

    For FastAPI and repeated predictions, create and reuse
    one DemandPredictor instance instead. This prevents the
    trained model from being loaded for every request.
    """

    predictor = DemandPredictor()

    return predictor.predict(
        demand_history=demand_history,
        weather_row=weather_row,
        target_timestamp=target_timestamp,
        zone_id=zone_id
    )