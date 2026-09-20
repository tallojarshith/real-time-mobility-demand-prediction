import pandas as pd

from src.features.demand import create_hourly_demand


def test_create_hourly_demand():

    # Small artificial trip dataset
    sample_data = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2025-01-01 00:10:00",  # Zone 1
            "2025-01-01 00:20:00",  # Zone 1
            "2025-01-01 01:15:00",  # Zone 2
            "2025-01-01 02:30:00",  # Zone 1
        ]),

        "PULocationID": [
            1,
            1,
            2,
            1
        ]
    })

    demand_df = create_hourly_demand(
        sample_data,
        start_date="2025-01-01 00:00:00",
        end_date="2025-01-01 03:00:00"
    )

    # 3 hours × 2 active zones
    assert len(demand_df) == 6

    # Total demand must equal original number of trips
    assert demand_df["demand"].sum() == len(sample_data)

    # No missing demand values
    assert demand_df["demand"].isna().sum() == 0

    # Check a known observed value:
    # Zone 1 had two pickups between 00:00–01:00
    zone1_hour0 = demand_df[
        (demand_df["pickup_hour_timestamp"]
         == pd.Timestamp("2025-01-01 00:00:00"))
        &
        (demand_df["PULocationID"] == 1)
    ]["demand"].iloc[0]

    assert zone1_hour0 == 2

    # Check a zero-demand combination:
    # Zone 2 had no pickups between 00:00–01:00
    zone2_hour0 = demand_df[
        (demand_df["pickup_hour_timestamp"]
         == pd.Timestamp("2025-01-01 00:00:00"))
        &
        (demand_df["PULocationID"] == 2)
    ]["demand"].iloc[0]

    assert zone2_hour0 == 0