import pandas as pd

from src.preprocessing.preprocess import clean_taxi_data
from src.features.demand import create_hourly_demand

from src.features.weather import (
    merge_weather_features,
    WEATHER_FEATURES
)

from src.features.feature_engineering import (
    add_time_features,
    add_demand_history_features
)

from src.utils.logger import get_logger


# =========================================================
# LOGGER
# =========================================================

logger = get_logger(__name__)


# =========================================================
# HISTORICAL DEMAND FEATURES
# =========================================================

HISTORY_FEATURES = [
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "rolling_std_24h"
]


# =========================================================
# MAIN FEATURE PIPELINE
# =========================================================

def build_feature_dataset(
    taxi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Build the complete zone-hour modelling dataset.

    Pipeline:
        Raw taxi trips
            -> target-aware cleaning
            -> hourly pickup aggregation
            -> complete zone-hour grid
            -> zero-demand representation
            -> weather merge
            -> calendar/cyclical features
            -> lag features
            -> rolling demand features
            -> remove rows without sufficient history

    Parameters
    ----------
    taxi_df : pd.DataFrame
        Raw NYC Yellow Taxi trip records.

    weather_df : pd.DataFrame
        Hourly NYC weather observations.

    start_date : str
        Beginning of the requested period.

    end_date : str
        End of the requested period.
        This value is exclusive.

    Returns
    -------
    pd.DataFrame
        Model-ready zone-hour feature dataset.
    """

    # =====================================================
    # START PIPELINE
    # =====================================================

    logger.info(
        "Starting feature dataset pipeline"
    )

    logger.info(
        "Requested period: %s to %s",
        start_date,
        end_date
    )

    logger.info(
        "Input taxi rows: %d | Weather rows: %d",
        len(taxi_df),
        len(weather_df)
    )

    # =====================================================
    # 1. CLEAN TAXI DATA
    # =====================================================

    logger.info(
        "Starting taxi data cleaning"
    )

    clean_df = clean_taxi_data(
        taxi_df,
        start_date=start_date,
        end_date=end_date
    )

    logger.info(
        "Taxi cleaning completed: %d valid trips",
        len(clean_df)
    )

    # =====================================================
    # 2. CREATE HOURLY TAXI DEMAND
    # =====================================================

    logger.info(
        "Creating hourly zone-level demand"
    )

    demand_df = create_hourly_demand(
        clean_df,
        start_date=start_date,
        end_date=end_date
    )

    logger.info(
        "Hourly demand dataset created: "
        "%d zone-hour rows",
        len(demand_df)
    )

    # =====================================================
    # 3. VALIDATE DEMAND AGGREGATION
    # =====================================================

    original_trip_count = len(clean_df)

    aggregated_trip_count = int(
        demand_df["demand"].sum()
    )

    if aggregated_trip_count != original_trip_count:

        logger.error(
            "Demand aggregation validation failed: "
            "clean trips=%d | aggregated demand=%d",
            original_trip_count,
            aggregated_trip_count
        )

        raise ValueError(
            "Aggregated demand does not match "
            "number of cleaned trips"
        )

    logger.info(
        "Demand aggregation validated: "
        "%d pickups preserved",
        aggregated_trip_count
    )

    # =====================================================
    # 4. MERGE WEATHER FEATURES
    # =====================================================

    logger.info(
        "Merging hourly weather features"
    )

    feature_df = merge_weather_features(
        demand_df,
        weather_df
    )

    # =====================================================
    # 5. VALIDATE WEATHER DATA
    # =====================================================

    missing_weather = int(
        feature_df[
            WEATHER_FEATURES
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_weather > 0:

        logger.error(
            "Weather validation failed: "
            "%d missing weather values",
            missing_weather
        )

        raise ValueError(
            f"Missing weather values after merge: "
            f"{missing_weather}"
        )

    logger.info(
        "Weather merge completed successfully"
    )

    # =====================================================
    # 6. CREATE TIME FEATURES
    # =====================================================

    logger.info(
        "Creating calendar and cyclical features"
    )

    feature_df = add_time_features(
        feature_df
    )

    logger.info(
        "Calendar and cyclical features created"
    )

    # =====================================================
    # 7. CREATE HISTORICAL DEMAND FEATURES
    # =====================================================

    logger.info(
        "Creating lag and rolling demand features"
    )

    feature_df = add_demand_history_features(
        feature_df
    )

    logger.info(
        "Lag and rolling demand features created"
    )

    # =====================================================
    # 8. REMOVE ROWS WITHOUT SUFFICIENT HISTORY
    # =====================================================

    rows_before_history_filter = len(
        feature_df
    )

    model_df = feature_df.dropna(
        subset=HISTORY_FEATURES
    ).reset_index(drop=True)

    rows_removed = (
        rows_before_history_filter
        - len(model_df)
    )

    logger.info(
        "Removed %d rows without sufficient "
        "historical context",
        rows_removed
    )

    # =====================================================
    # 9. FINAL HISTORY VALIDATION
    # =====================================================

    remaining_missing_history = int(
        model_df[
            HISTORY_FEATURES
        ]
        .isna()
        .sum()
        .sum()
    )

    if remaining_missing_history > 0:

        logger.error(
            "Historical feature validation failed: "
            "%d missing values remain",
            remaining_missing_history
        )

        raise ValueError(
            "Missing historical features remain "
            "after history filtering"
        )

    # =====================================================
    # PIPELINE COMPLETE
    # =====================================================

    logger.info(
        "Feature pipeline completed successfully: "
        "%d model-ready rows",
        len(model_df)
    )

    return model_df