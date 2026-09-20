import numpy as np
import pandas as pd

from src.features.pipeline import (
    build_feature_dataset,
    HISTORY_FEATURES
)

from src.features.weather import WEATHER_FEATURES


def test_build_feature_dataset():

    # --------------------------------------------------
    # Create 200 hours of synthetic taxi trips
    # --------------------------------------------------

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=200,
        freq="h"
    )

    taxi_rows = []

    # Two taxi zones.
    # Each zone receives one pickup every hour.
    for timestamp in timestamps:

        taxi_rows.append({
            "tpep_pickup_datetime":
                timestamp + pd.Timedelta(minutes=10),
            "PULocationID": 161
        })

        taxi_rows.append({
            "tpep_pickup_datetime":
                timestamp + pd.Timedelta(minutes=20),
            "PULocationID": 132
        })

    taxi_df = pd.DataFrame(taxi_rows)

    # --------------------------------------------------
    # Create matching hourly weather
    # --------------------------------------------------

    weather_df = pd.DataFrame({
        "time": timestamps,

        "temperature_2m":
            np.full(len(timestamps), 5.0),

        "relative_humidity_2m":
            np.full(len(timestamps), 80),

        "precipitation":
            np.zeros(len(timestamps)),

        "rain":
            np.zeros(len(timestamps)),

        "snowfall":
            np.zeros(len(timestamps)),

        "weather_code":
            np.ones(len(timestamps)),

        "wind_speed_10m":
            np.full(len(timestamps), 10.0)
    })

    # --------------------------------------------------
    # Run complete pipeline
    # --------------------------------------------------

    model_df = build_feature_dataset(
        taxi_df=taxi_df,
        weather_df=weather_df,
        start_date="2025-01-01 00:00:00",
        end_date="2025-01-09 08:00:00"
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    # Original grid:
    # 200 hours × 2 zones = 400 rows
    #
    # First 168 hours per zone cannot have lag_168h:
    # 168 × 2 = 336 removed
    #
    # Remaining:
    # 400 - 336 = 64 rows

    assert len(model_df) == 64

    # Historical features must contain no missing values
    assert (
        model_df[HISTORY_FEATURES]
        .isna()
        .sum()
        .sum()
        == 0
    )

    # Weather features must contain no missing values
    assert (
        model_df[WEATHER_FEATURES]
        .isna()
        .sum()
        .sum()
        == 0
    )

    # Every synthetic zone-hour has exactly one pickup
    assert (model_df["demand"] == 1).all()

    # Required time features should exist
    required_time_features = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos"
    ]

    for feature in required_time_features:
        assert feature in model_df.columns