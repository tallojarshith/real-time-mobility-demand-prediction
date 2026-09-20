import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and cyclical time features.
    """

    df = df.copy()

    timestamp = df["pickup_hour_timestamp"]

    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["day_of_month"] = timestamp.dt.day
    df["month"] = timestamp.dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclical weekday encoding
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def add_demand_history_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create leakage-safe historical demand features.

    All historical features are shifted before rolling
    calculations so the current hour's target is never
    included in its own predictors.
    """

    df = df.copy()

    # Critical: chronological ordering within each zone
    df = df.sort_values(
        ["PULocationID", "pickup_hour_timestamp"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        "PULocationID"
    )["demand"]

    # Historical lags
    df["lag_1h"] = grouped.shift(1)
    df["lag_24h"] = grouped.shift(24)
    df["lag_168h"] = grouped.shift(168)

    # Previous demand only — excludes current target
    previous_demand = grouped.shift(1)

    # Rolling statistics calculated independently per zone
    df["rolling_mean_3h"] = (
        previous_demand
        .groupby(df["PULocationID"])
        .rolling(window=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_24h"] = (
        previous_demand
        .groupby(df["PULocationID"])
        .rolling(window=24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_std_24h"] = (
        previous_demand
        .groupby(df["PULocationID"])
        .rolling(window=24)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df