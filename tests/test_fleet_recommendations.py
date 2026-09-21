import pandas as pd
import pytest

from src.fleet.recommendations import (
    add_zone_metadata,
    create_fleet_recommendations
)


# =========================================================
# TEST 1: ZONE METADATA
# =========================================================

def test_add_zone_metadata():

    predictions = pd.DataFrame({
        "pickup_hour_timestamp": [
            "2025-01-28 18:00:00",
            "2025-01-28 18:00:00"
        ],
        "PULocationID": [
            161,
            132
        ],
        "predicted_demand": [
            200.0,
            150.0
        ]
    })

    zone_lookup = pd.DataFrame({
        "LocationID": [
            161,
            132
        ],
        "Borough": [
            "Manhattan",
            "Queens"
        ],
        "Zone": [
            "Midtown Center",
            "JFK Airport"
        ],
        "service_zone": [
            "Yellow Zone",
            "Airports"
        ]
    })

    result = add_zone_metadata(
        predictions=predictions,
        zone_lookup=zone_lookup
    )

    assert len(result) == 2

    assert "Zone" in result.columns
    assert "Borough" in result.columns
    assert "service_zone" in result.columns

    midtown = result[
        result["PULocationID"] == 161
    ].iloc[0]

    assert midtown["Zone"] == "Midtown Center"
    assert midtown["Borough"] == "Manhattan"


# =========================================================
# TEST 2: FLEET ALLOCATION
# =========================================================

def test_create_fleet_recommendations():

    predictions = pd.DataFrame({
        "PULocationID": [
            161,
            132,
            79,
            237
        ],
        "predicted_demand": [
            200.0,
            150.0,
            100.0,
            50.0
        ]
    })

    result = create_fleet_recommendations(
        predictions=predictions,
        available_vehicles=100
    )

    # Every available vehicle must be allocated
    assert (
        result["recommended_vehicles"].sum()
        == 100
    )

    # No negative allocation
    assert (
        result["recommended_vehicles"] >= 0
    ).all()

    # Demand shares should sum to 1
    assert abs(
        result["demand_share"].sum() - 1.0
    ) < 1e-9

    # Highest-demand zone should receive most vehicles
    midtown = result[
        result["PULocationID"] == 161
    ].iloc[0]

    assert (
        midtown["recommended_vehicles"]
        == 40
    )


# =========================================================
# TEST 3: ROUNDING MUST PRESERVE FLEET SIZE
# =========================================================

def test_allocation_preserves_vehicle_count():

    predictions = pd.DataFrame({
        "PULocationID": [
            1,
            2,
            3
        ],
        "predicted_demand": [
            1.0,
            1.0,
            1.0
        ]
    })

    result = create_fleet_recommendations(
        predictions=predictions,
        available_vehicles=10
    )

    # 10 vehicles across equal-demand zones might become
    # 4,3,3 after largest-remainder allocation.
    assert (
        result["recommended_vehicles"].sum()
        == 10
    )


# =========================================================
# TEST 4: INVALID VEHICLE COUNT
# =========================================================

def test_invalid_vehicle_count():

    predictions = pd.DataFrame({
        "PULocationID": [161],
        "predicted_demand": [100.0]
    })

    with pytest.raises(
        ValueError,
        match="available_vehicles must be greater than 0"
    ):

        create_fleet_recommendations(
            predictions=predictions,
            available_vehicles=0
        )


# =========================================================
# TEST 5: ZERO TOTAL DEMAND
# =========================================================

def test_zero_total_demand():

    predictions = pd.DataFrame({
        "PULocationID": [
            161,
            132
        ],
        "predicted_demand": [
            0.0,
            0.0
        ]
    })

    with pytest.raises(
        ValueError,
        match="Total predicted demand must be greater than 0"
    ):

        create_fleet_recommendations(
            predictions=predictions,
            available_vehicles=100
        )