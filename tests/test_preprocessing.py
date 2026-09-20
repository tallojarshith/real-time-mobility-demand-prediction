import pandas as pd
import pytest

from src.preprocessing.preprocess import clean_taxi_data


def test_clean_taxi_data():
    """
    Test that taxi preprocessing:

    1. Keeps valid trips inside the requested date range.
    2. Removes trips outside the requested date range.
    3. Removes invalid taxi-zone IDs.
    4. Removes rows with invalid/missing pickup timestamps.
    """

    # -----------------------------------------------------
    # Create synthetic raw taxi data
    # -----------------------------------------------------

    df = pd.DataFrame({
        "tpep_pickup_datetime": [
            "2025-01-15 10:00:00",   # Valid
            "2024-12-31 23:59:00",   # Before start date
            "2025-02-01 00:00:00",   # End date - excluded
            "2025-01-20 12:00:00",   # Invalid zone
            None                     # Missing timestamp
        ],

        "PULocationID": [
            161,
            161,
            161,
            300,
            161
        ]
    })

    # -----------------------------------------------------
    # Run preprocessing
    # -----------------------------------------------------

    clean_df = clean_taxi_data(
        df=df,
        start_date="2025-01-01",
        end_date="2025-02-01"
    )

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------

    # Only the first row should survive
    assert len(clean_df) == 1

    # Verify correct taxi zone
    assert clean_df.iloc[0]["PULocationID"] == 161

    # Verify correct timestamp
    assert (
        clean_df.iloc[0]["tpep_pickup_datetime"]
        == pd.Timestamp("2025-01-15 10:00:00")
    )

    # Verify all remaining zones are valid
    assert clean_df["PULocationID"].between(
        1,
        265
    ).all()

    # Verify there are no missing pickup timestamps
    assert (
        clean_df["tpep_pickup_datetime"]
        .isna()
        .sum()
        == 0
    )


def test_invalid_date_range():
    """
    start_date must be earlier than end_date.
    """

    df = pd.DataFrame({
        "tpep_pickup_datetime": [
            "2025-01-15 10:00:00"
        ],
        "PULocationID": [
            161
        ]
    })

    with pytest.raises(
        ValueError,
        match="start_date must be earlier than end_date"
    ):
        clean_taxi_data(
            df=df,
            start_date="2025-02-01",
            end_date="2025-01-01"
        )


def test_missing_required_columns():
    """
    The preprocessing function should fail clearly
    when required input columns are missing.
    """

    df = pd.DataFrame({
        "some_other_column": [
            1,
            2,
            3
        ]
    })

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        clean_taxi_data(
            df=df,
            start_date="2025-01-01",
            end_date="2025-02-01"
        )