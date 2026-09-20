import numpy as np
import pandas as pd

from src.prediction.inference_features import (
    build_inference_features,
    MODEL_FEATURES
)


def test_build_inference_features():

    # -----------------------------------------------------
    # Historical demand
    # -----------------------------------------------------

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=200,
        freq="h"
    )

    demand_history = pd.DataFrame({
        "pickup_hour_timestamp": timestamps,
        "PULocationID": 161,
        "demand": np.arange(1, 201)
    })

    # Predict hour immediately after our history
    target_timestamp = pd.Timestamp(
        "2025-01-09 08:00:00"
    )

    # -----------------------------------------------------
    # Target-hour weather forecast
    # -----------------------------------------------------

    weather_row = pd.DataFrame({
        "temperature_2m": [7.5],
        "relative_humidity_2m": [75],
        "precipitation": [0.0],
        "rain": [0.0],
        "snowfall": [0.0],
        "weather_code": [1],
        "wind_speed_10m": [12.0]
    })

    # -----------------------------------------------------
    # Build inference features
    # -----------------------------------------------------

    features = build_inference_features(
        demand_history=demand_history,
        weather_row=weather_row,
        target_timestamp=target_timestamp,
        zone_id=161
    )

    # -----------------------------------------------------
    # Validate schema
    # -----------------------------------------------------

    assert features.shape == (1, 22)

    assert list(features.columns) == MODEL_FEATURES

    assert features.isna().sum().sum() == 0

    # -----------------------------------------------------
    # Validate lag calculations
    # -----------------------------------------------------

    # Target = hour 200.
    # Previous hour contains demand 200.
    assert features.iloc[0]["lag_1h"] == 200

    # 24 hours before target corresponds to demand 177.
    assert features.iloc[0]["lag_24h"] == 177

    # 168 hours before target corresponds to demand 33.
    assert features.iloc[0]["lag_168h"] == 33

    # -----------------------------------------------------
    # Validate rolling calculations
    # -----------------------------------------------------

    expected_3h_mean = np.mean([
        198,
        199,
        200
    ])

    assert np.isclose(
        features.iloc[0]["rolling_mean_3h"],
        expected_3h_mean
    )

    expected_24h = np.arange(
        177,
        201
    )

    assert np.isclose(
        features.iloc[0]["rolling_mean_24h"],
        expected_24h.mean()
    )

    assert np.isclose(
        features.iloc[0]["rolling_std_24h"],
        expected_24h.std(ddof=1)
    )


def test_future_demand_is_not_used():

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=201,
        freq="h"
    )

    demand_values = np.arange(
        1,
        202,
        dtype=float
    )

    # Make target-hour demand absurdly large.
    # It MUST NOT influence our features.
    demand_values[-1] = 999999

    demand_history = pd.DataFrame({
        "pickup_hour_timestamp": timestamps,
        "PULocationID": 161,
        "demand": demand_values
    })

    target_timestamp = timestamps[-1]

    weather_row = pd.DataFrame({
        "temperature_2m": [7.5],
        "relative_humidity_2m": [75],
        "precipitation": [0.0],
        "rain": [0.0],
        "snowfall": [0.0],
        "weather_code": [1],
        "wind_speed_10m": [12.0]
    })

    features = build_inference_features(
        demand_history=demand_history,
        weather_row=weather_row,
        target_timestamp=target_timestamp,
        zone_id=161
    )

    # Previous observed hour should be 200.
    assert features.iloc[0]["lag_1h"] == 200

    # Future/current target demand 999999 must not appear
    # anywhere in the historical-demand features.
    history_features = [
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_3h",
        "rolling_mean_24h",
        "rolling_std_24h"
    ]

    assert not (
        features[
            history_features
        ] == 999999
    ).any().any()