import pandas as pd


WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m"
]


def merge_weather_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge hourly NYC weather data with zone-hour taxi demand.

    Weather is city-level, so the same hourly weather observation
    is shared by all taxi zones for that hour.
    """

    demand_df = demand_df.copy()
    weather_df = weather_df.copy()

    # Validate demand dataframe
    if "pickup_hour_timestamp" not in demand_df.columns:
        raise ValueError(
            "demand_df must contain pickup_hour_timestamp"
        )

    # Validate weather dataframe
    required_weather_columns = [
        "time",
        *WEATHER_FEATURES
    ]

    missing_columns = [
        col for col in required_weather_columns
        if col not in weather_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing weather columns: {missing_columns}"
        )

    # Convert weather timestamp
    weather_df["pickup_hour_timestamp"] = pd.to_datetime(
        weather_df["time"],
        errors="coerce"
    )

    # Invalid weather timestamps should not enter pipeline
    weather_df = weather_df.dropna(
        subset=["pickup_hour_timestamp"]
    )

    # Prevent duplicate hourly weather records
    if weather_df["pickup_hour_timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate hourly timestamps found in weather data"
        )

    rows_before = len(demand_df)

    merged = demand_df.merge(
        weather_df[
            ["pickup_hour_timestamp", *WEATHER_FEATURES]
        ],
        on="pickup_hour_timestamp",
        how="left"
    )

    # Merge must never create/remove demand rows
    if len(merged) != rows_before:
        raise ValueError(
            "Weather merge changed the number of demand rows"
        )

    return merged