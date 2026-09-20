import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    add_time_features,
    add_demand_history_features
)


def test_time_features():

    df = pd.DataFrame({
        "pickup_hour_timestamp": pd.to_datetime([
            "2025-01-06 00:00:00",  # Monday
            "2025-01-11 12:00:00"   # Saturday
        ])
    })

    result = add_time_features(df)

    # Monday = 0
    assert result.loc[0, "day_of_week"] == 0

    # Saturday = 5
    assert result.loc[1, "day_of_week"] == 5

    assert result.loc[0, "is_weekend"] == 0
    assert result.loc[1, "is_weekend"] == 1

    # Midnight cyclical encoding
    assert np.isclose(result.loc[0, "hour_sin"], 0.0)
    assert np.isclose(result.loc[0, "hour_cos"], 1.0)


def test_demand_history_features():

    # 200 hours gives enough history to test lag_168h
    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=200,
        freq="h"
    )

    # Easy-to-check demand:
    # 1, 2, 3, 4, ...
    df = pd.DataFrame({
        "pickup_hour_timestamp": timestamps,
        "PULocationID": 161,
        "demand": np.arange(1, 201)
    })

    result = add_demand_history_features(df)

    # Row 168 has current demand = 169

    assert result.loc[168, "lag_1h"] == 168

    assert result.loc[168, "lag_24h"] == 145

    assert result.loc[168, "lag_168h"] == 1

    # Previous 3 demands are:
    # 166, 167, 168
    expected_mean_3h = np.mean([
        166, 167, 168
    ])

    assert np.isclose(
        result.loc[168, "rolling_mean_3h"],
        expected_mean_3h
    )

    # Previous 24 values:
    # demand 145 through 168
    expected_mean_24h = np.mean(
        np.arange(145, 169)
    )

    assert np.isclose(
        result.loc[168, "rolling_mean_24h"],
        expected_mean_24h
    )

    # Most important leakage check:
    # current demand 169 must NOT be included
    assert (
        result.loc[168, "rolling_mean_3h"]
        != np.mean([167, 168, 169])
    )