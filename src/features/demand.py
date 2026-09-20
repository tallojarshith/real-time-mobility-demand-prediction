import pandas as pd


def create_hourly_demand(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Convert cleaned taxi trip records into hourly pickup
    demand for each active taxi zone.

    A complete zone-hour grid is created so that hours with
    no pickups are represented explicitly as demand = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned taxi trip-level dataframe.

    start_date : str
        Start timestamp of the period.

    end_date : str
        End timestamp (exclusive).

    Returns
    -------
    pd.DataFrame
        Columns:
        pickup_hour_timestamp
        PULocationID
        demand
    """

    required_columns = [
        "tpep_pickup_datetime",
        "PULocationID"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    # Convert each pickup timestamp to its hour
    df["pickup_hour_timestamp"] = (
        df["tpep_pickup_datetime"].dt.floor("h")
    )

    # Count actual pickups for each zone-hour
    observed_demand = (
        df.groupby(
            ["pickup_hour_timestamp", "PULocationID"]
        )
        .size()
        .reset_index(name="demand")
    )

    # Zones actually present in the data
    active_zones = sorted(
        df["PULocationID"].unique()
    )

    # Complete hourly timeline
    all_hours = pd.date_range(
        start=start_date,
        end=end_date,
        freq="h",
        inclusive="left"
    )

    # Create every possible hour × active-zone combination
    full_grid = pd.MultiIndex.from_product(
        [all_hours, active_zones],
        names=[
            "pickup_hour_timestamp",
            "PULocationID"
        ]
    ).to_frame(index=False)

    # Add observed demand to complete grid
    demand_df = full_grid.merge(
        observed_demand,
        on=[
            "pickup_hour_timestamp",
            "PULocationID"
        ],
        how="left"
    )

    # Missing combination means zero pickups
    demand_df["demand"] = (
        demand_df["demand"]
        .fillna(0)
        .astype("int32")
    )

    return demand_df