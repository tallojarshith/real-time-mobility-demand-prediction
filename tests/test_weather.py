import pandas as pd
import pytest

from src.features.weather import merge_weather_features


def test_merge_weather_features():

    demand_df = pd.DataFrame({
        "pickup_hour_timestamp": pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 01:00:00"
        ]),
        "PULocationID": [1, 2, 1, 2],
        "demand": [5, 10, 7, 12]
    })

    weather_df = pd.DataFrame({
        "time": [
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00"
        ],
        "temperature_2m": [5.0, 4.5],
        "relative_humidity_2m": [80, 82],
        "precipitation": [0.0, 0.2],
        "rain": [0.0, 0.2],
        "snowfall": [0.0, 0.0],
        "weather_code": [1, 61],
        "wind_speed_10m": [10.0, 12.0]
    })

    result = merge_weather_features(
        demand_df,
        weather_df
    )

    # Merge must preserve demand rows
    assert len(result) == len(demand_df)

    # Every row should have weather
    assert result["temperature_2m"].isna().sum() == 0

    # Both taxi zones at 00:00 should receive
    # the same city-level weather
    hour_zero = result[
        result["pickup_hour_timestamp"]
        == pd.Timestamp("2025-01-01 00:00:00")
    ]

    assert len(hour_zero) == 2

    assert (
        hour_zero["temperature_2m"] == 5.0
    ).all()


def test_duplicate_weather_timestamp():

    demand_df = pd.DataFrame({
        "pickup_hour_timestamp": pd.to_datetime([
            "2025-01-01 00:00:00"
        ]),
        "PULocationID": [1],
        "demand": [5]
    })

    # Deliberately duplicate the same weather hour
    weather_df = pd.DataFrame({
        "time": [
            "2025-01-01 00:00:00",
            "2025-01-01 00:00:00"
        ],
        "temperature_2m": [5.0, 6.0],
        "relative_humidity_2m": [80, 81],
        "precipitation": [0.0, 0.0],
        "rain": [0.0, 0.0],
        "snowfall": [0.0, 0.0],
        "weather_code": [1, 1],
        "wind_speed_10m": [10.0, 11.0]
    })

    with pytest.raises(
        ValueError,
        match="Duplicate hourly timestamps"
    ):
        merge_weather_features(
            demand_df,
            weather_df
        )