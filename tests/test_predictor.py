import numpy as np
import pandas as pd

from src.prediction.predictor import DemandPredictor


# =========================================================
# TEST 1: SINGLE-ZONE PREDICTION
# =========================================================

def test_demand_predictor():
    """
    Test end-to-end prediction for a single taxi zone.

    Flow:
        historical demand
            -> inference features
            -> trained Extra Trees model
            -> demand prediction
    """

    # -----------------------------------------------------
    # Create 200 hours of historical demand
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

    # Predict the hour immediately after history
    target_timestamp = pd.Timestamp(
        "2025-01-09 08:00:00"
    )

    # -----------------------------------------------------
    # Weather for prediction hour
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
    # Initialize predictor
    # -----------------------------------------------------

    predictor = DemandPredictor()

    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = predictor.predict(
        demand_history=demand_history,
        weather_row=weather_row,
        target_timestamp=target_timestamp,
        zone_id=161
    )

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------

    assert isinstance(
        prediction,
        (float, np.floating)
    )

    assert np.isfinite(
        prediction
    )

    # Taxi demand cannot be negative
    assert prediction >= 0


# =========================================================
# TEST 2: MULTI-ZONE PREDICTION
# =========================================================

def test_multiple_zone_prediction():
    """
    Test batch prediction for multiple NYC taxi zones.
    """

    # -----------------------------------------------------
    # Create 200 hours of historical demand
    # for two taxi zones
    # -----------------------------------------------------

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=200,
        freq="h"
    )

    rows = []

    for zone_id in [161, 132]:

        for i, timestamp in enumerate(
            timestamps,
            start=1
        ):

            rows.append({
                "pickup_hour_timestamp": timestamp,
                "PULocationID": zone_id,
                "demand": i
            })

    demand_history = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # Target prediction hour
    # -----------------------------------------------------

    target_timestamp = pd.Timestamp(
        "2025-01-09 08:00:00"
    )

    # -----------------------------------------------------
    # Weather for prediction hour
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
    # Initialize predictor
    # -----------------------------------------------------

    predictor = DemandPredictor()

    # -----------------------------------------------------
    # Generate predictions for both zones
    # -----------------------------------------------------

    results = predictor.predict_multiple_zones(
        demand_history=demand_history,
        weather_row=weather_row,
        target_timestamp=target_timestamp,
        zone_ids=[161, 132]
    )

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------

    # We requested two zones
    assert len(results) == 2

    # Both zone IDs should be returned
    assert set(
        results["PULocationID"]
    ) == {161, 132}

    # Predictions cannot be negative
    assert (
        results["predicted_demand"] >= 0
    ).all()

    # Predictions cannot contain NaN
    assert (
        results["predicted_demand"]
        .isna()
        .sum()
        == 0
    )

    # Prediction timestamp should be correct
    assert (
        results["pickup_hour_timestamp"]
        == target_timestamp
    ).all()

    # Results should be sorted from
    # highest predicted demand to lowest
    assert (
        results["predicted_demand"]
        .is_monotonic_decreasing
    )