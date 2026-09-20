import numpy as np
import pandas as pd

from src.features.weather import WEATHER_FEATURES
from src.utils.logger import get_logger


logger = get_logger(__name__)


MODEL_FEATURES = [
    "PULocationID",
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "rolling_std_24h"
]


def build_inference_features(
    demand_history: pd.DataFrame,
    weather_row: pd.DataFrame,
    target_timestamp,
    zone_id: int
) -> pd.DataFrame:
    """
    Build the 22 model features required to predict demand
    for one taxi zone at one future hour.

    Only demand observed BEFORE target_timestamp is used.

    Parameters
    ----------
    demand_history : pd.DataFrame
        Historical zone-hour demand containing:
        pickup_hour_timestamp, PULocationID, demand

    weather_row : pd.DataFrame
        Weather information for the target prediction hour.

    target_timestamp :
        Hour for which demand should be predicted.

    zone_id : int
        NYC taxi-zone ID.

    Returns
    -------
    pd.DataFrame
        One-row dataframe containing the exact 22 features
        required by the trained model.
    """

    logger.info(
        "Building inference features for zone=%s timestamp=%s",
        zone_id,
        target_timestamp
    )

    target_timestamp = pd.Timestamp(
        target_timestamp
    )

    # =====================================================
    # 1. VALIDATE DEMAND HISTORY
    # =====================================================

    required_history_columns = [
        "pickup_hour_timestamp",
        "PULocationID",
        "demand"
    ]

    missing_columns = [
        column
        for column in required_history_columns
        if column not in demand_history.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing demand history columns: {missing_columns}"
        )

    history = demand_history.copy()

    history["pickup_hour_timestamp"] = pd.to_datetime(
        history["pickup_hour_timestamp"],
        errors="coerce"
    )

    history = history.dropna(
        subset=["pickup_hour_timestamp"]
    )

    # Keep only requested zone
    zone_history = history[
        history["PULocationID"] == zone_id
    ].copy()

    # CRITICAL:
    # Never use demand from target_timestamp or the future.
    zone_history = zone_history[
        zone_history["pickup_hour_timestamp"]
        < target_timestamp
    ]

    zone_history = zone_history.sort_values(
        "pickup_hour_timestamp"
    )

    # Prevent duplicate zone-hour demand records
    if zone_history[
        "pickup_hour_timestamp"
    ].duplicated().any():

        raise ValueError(
            "Duplicate timestamps found in zone demand history"
        )

    if zone_history.empty:
        raise ValueError(
            f"No historical demand found for zone {zone_id}"
        )

    # =====================================================
    # 2. CREATE TIMESTAMP LOOKUP
    # =====================================================

    demand_lookup = zone_history.set_index(
        "pickup_hour_timestamp"
    )["demand"]

    # =====================================================
    # 3. CREATE LAG FEATURES
    # =====================================================

    lag_1_timestamp = (
        target_timestamp
        - pd.Timedelta(hours=1)
    )

    lag_24_timestamp = (
        target_timestamp
        - pd.Timedelta(hours=24)
    )

    lag_168_timestamp = (
        target_timestamp
        - pd.Timedelta(hours=168)
    )

    required_lag_timestamps = {
        "lag_1h": lag_1_timestamp,
        "lag_24h": lag_24_timestamp,
        "lag_168h": lag_168_timestamp
    }

    lag_values = {}

    for feature_name, timestamp in (
        required_lag_timestamps.items()
    ):

        if timestamp not in demand_lookup.index:

            logger.error(
                "Missing historical demand required for %s "
                "at %s",
                feature_name,
                timestamp
            )

            raise ValueError(
                f"Missing demand required for "
                f"{feature_name}: {timestamp}"
            )

        lag_values[feature_name] = float(
            demand_lookup.loc[timestamp]
        )

    # =====================================================
    # 4. CREATE ROLLING FEATURES
    # =====================================================

    previous_3_hours = pd.date_range(
        end=target_timestamp - pd.Timedelta(hours=1),
        periods=3,
        freq="h"
    )

    previous_24_hours = pd.date_range(
        end=target_timestamp - pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )

    missing_3h = previous_3_hours.difference(
        demand_lookup.index
    )

    if len(missing_3h) > 0:
        raise ValueError(
            "Insufficient continuous history for "
            "rolling_mean_3h"
        )

    missing_24h = previous_24_hours.difference(
        demand_lookup.index
    )

    if len(missing_24h) > 0:
        raise ValueError(
            "Insufficient continuous history for "
            "24-hour rolling features"
        )

    demand_3h = demand_lookup.loc[
        previous_3_hours
    ]

    demand_24h = demand_lookup.loc[
        previous_24_hours
    ]

    rolling_mean_3h = float(
        demand_3h.mean()
    )

    rolling_mean_24h = float(
        demand_24h.mean()
    )

    # pandas rolling std() uses sample standard deviation
    # with ddof=1 by default.
    rolling_std_24h = float(
        demand_24h.std(ddof=1)
    )

    # =====================================================
    # 5. CREATE CALENDAR FEATURES
    # =====================================================

    hour = target_timestamp.hour
    day_of_week = target_timestamp.dayofweek
    day_of_month = target_timestamp.day
    is_weekend = int(day_of_week >= 5)

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    dow_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    dow_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # =====================================================
    # 6. VALIDATE TARGET-HOUR WEATHER
    # =====================================================

    if weather_row.empty:
        raise ValueError(
            "Weather data is empty for target hour"
        )

    if len(weather_row) != 1:
        raise ValueError(
            "weather_row must contain exactly one row"
        )

    missing_weather_columns = [
        column
        for column in WEATHER_FEATURES
        if column not in weather_row.columns
    ]

    if missing_weather_columns:
        raise ValueError(
            f"Missing weather columns: "
            f"{missing_weather_columns}"
        )

    if weather_row[
        WEATHER_FEATURES
    ].isna().any().any():

        raise ValueError(
            "Weather data contains missing values"
        )

    weather = weather_row.iloc[0]

    # =====================================================
    # 7. BUILD MODEL INPUT
    # =====================================================

    feature_data = {
        "PULocationID": zone_id,

        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "is_weekend": is_weekend,

        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,

        "temperature_2m":
            weather["temperature_2m"],

        "relative_humidity_2m":
            weather["relative_humidity_2m"],

        "precipitation":
            weather["precipitation"],

        "rain":
            weather["rain"],

        "snowfall":
            weather["snowfall"],

        "weather_code":
            weather["weather_code"],

        "wind_speed_10m":
            weather["wind_speed_10m"],

        "lag_1h":
            lag_values["lag_1h"],

        "lag_24h":
            lag_values["lag_24h"],

        "lag_168h":
            lag_values["lag_168h"],

        "rolling_mean_3h":
            rolling_mean_3h,

        "rolling_mean_24h":
            rolling_mean_24h,

        "rolling_std_24h":
            rolling_std_24h
    }

    feature_df = pd.DataFrame(
        [feature_data]
    )

    # Enforce exact trained feature order
    feature_df = feature_df[
        MODEL_FEATURES
    ]

    if feature_df.isna().any().any():
        raise ValueError(
            "Inference features contain missing values"
        )

    logger.info(
        "Inference features successfully created "
        "for zone=%s",
        zone_id
    )

    return feature_df