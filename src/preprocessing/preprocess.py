import pandas as pd


def clean_taxi_data(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Clean NYC Yellow Taxi trip data for a requested time period.

    Cleaning is target-aware because our target is pickup demand.

    A trip is retained when:
    1. Required columns are available.
    2. Pickup timestamp is valid.
    3. Pickup timestamp falls within [start_date, end_date).
    4. Pickup taxi-zone ID is between 1 and 265.

    Missing passenger, fare, distance, payment, or other
    trip-related fields do not automatically invalidate a pickup.

    Parameters
    ----------
    df : pd.DataFrame
        Raw NYC Yellow Taxi trip dataframe.

    start_date : str
        Start of the required time period.

    end_date : str
        End of the required time period (exclusive).

    Returns
    -------
    pd.DataFrame
        Cleaned taxi trip dataframe.
    """

    # =====================================================
    # 1. VALIDATE REQUIRED COLUMNS
    # =====================================================

    required_columns = [
        "tpep_pickup_datetime",
        "PULocationID"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work on a copy so the original dataframe
    # is not modified.
    df = df.copy()

    # =====================================================
    # 2. CONVERT PICKUP TIMESTAMP
    # =====================================================

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        errors="coerce"
    )

    # Invalid timestamps become NaT.
    # These rows cannot be used for demand aggregation.
    df = df.dropna(
        subset=["tpep_pickup_datetime"]
    )

    # =====================================================
    # 3. CONVERT INPUT DATES
    # =====================================================

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date >= end_date:
        raise ValueError(
            "start_date must be earlier than end_date"
        )

    # =====================================================
    # 4. FILTER REQUESTED TIME PERIOD
    # =====================================================

    df = df[
        (df["tpep_pickup_datetime"] >= start_date)
        &
        (df["tpep_pickup_datetime"] < end_date)
    ]

    # =====================================================
    # 5. FILTER VALID TAXI ZONES
    # =====================================================

    # NYC TLC taxi-zone IDs are expected to be
    # between 1 and 265.
    df = df[
        df["PULocationID"].between(
            1,
            265,
            inclusive="both"
        )
    ]

    # =====================================================
    # 6. RESET INDEX
    # =====================================================

    df = df.reset_index(drop=True)

    return df