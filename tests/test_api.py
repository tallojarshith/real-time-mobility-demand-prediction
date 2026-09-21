import pandas as pd

from fastapi.testclient import TestClient

from api.main import app


# =========================================================
# HELPER: SINGLE-ZONE PAYLOAD
# =========================================================

def create_prediction_payload():

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=200,
        freq="h"
    )

    demand_history = []

    for i, timestamp in enumerate(
        timestamps,
        start=1
    ):

        demand_history.append({
            "pickup_hour_timestamp":
                timestamp.isoformat(),

            "PULocationID":
                161,

            "demand":
                float(i)
        })

    return {
        "target_timestamp":
            "2025-01-09T08:00:00",

        "zone_id":
            161,

        "demand_history":
            demand_history,

        "weather": {
            "temperature_2m": 7.5,
            "relative_humidity_2m": 75,
            "precipitation": 0.0,
            "rain": 0.0,
            "snowfall": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 12.0
        }
    }


# =========================================================
# HELPER: MULTI-ZONE PAYLOAD
# =========================================================

def create_fleet_payload():

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=200,
        freq="h"
    )

    demand_history = []

    # Two zones with different demand patterns
    for zone_id in [
        161,
        132
    ]:

        for i, timestamp in enumerate(
            timestamps,
            start=1
        ):

            demand_history.append({
                "pickup_hour_timestamp":
                    timestamp.isoformat(),

                "PULocationID":
                    zone_id,

                "demand":
                    float(i)
            })

    return {
        "target_timestamp":
            "2025-01-09T08:00:00",

        "available_vehicles":
            100,

        "zone_ids": [
            161,
            132
        ],

        "demand_history":
            demand_history,

        "weather": {
            "temperature_2m": 7.5,
            "relative_humidity_2m": 75,
            "precipitation": 0.0,
            "rain": 0.0,
            "snowfall": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 12.0
        }
    }


# =========================================================
# TEST 1: INFRASTRUCTURE ENDPOINTS
# =========================================================

def test_api_infrastructure():

    with TestClient(app) as client:

        # Root
        root_response = client.get(
            "/"
        )

        assert (
            root_response.status_code
            == 200
        )

        root_data = (
            root_response.json()
        )

        assert (
            root_data["version"]
            == "1.0.0"
        )

        # Health
        health_response = client.get(
            "/health"
        )

        assert (
            health_response.status_code
            == 200
        )

        health = (
            health_response.json()
        )

        assert (
            health["status"]
            == "healthy"
        )

        assert (
            health["model_loaded"]
            is True
        )

        assert (
            health["feature_count"]
            == 22
        )

        # Model info
        model_response = client.get(
            "/model/info"
        )

        assert (
            model_response.status_code
            == 200
        )

        model_info = (
            model_response.json()
        )

        assert (
            model_info["model_type"]
            == "ExtraTreesRegressor"
        )

        assert (
            model_info["feature_count"]
            == 22
        )

        assert (
            model_info[
                "forecast_horizon"
            ]
            == "1 hour"
        )


# =========================================================
# TEST 2: SINGLE-ZONE PREDICTION
# =========================================================

def test_prediction_endpoint():

    payload = (
        create_prediction_payload()
    )

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload
        )

        assert (
            response.status_code
            == 200
        )

        result = (
            response.json()
        )

        assert (
            result["zone_id"]
            == 161
        )

        assert (
            result[
                "predicted_demand"
            ]
            >= 0
        )

        assert (
            result[
                "target_timestamp"
            ].startswith(
                "2025-01-09T08:00:00"
            )
        )


# =========================================================
# TEST 3: INVALID ZONE
# =========================================================

def test_invalid_zone_request():

    payload = (
        create_prediction_payload()
    )

    # NYC TLC zone IDs are 1-265
    payload["zone_id"] = 300

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload
        )

        # FastAPI/Pydantic validation error
        assert (
            response.status_code
            == 422
        )


# =========================================================
# TEST 4: INSUFFICIENT HISTORY
# =========================================================

def test_insufficient_history():

    payload = (
        create_prediction_payload()
    )

    # Keep only 10 hours.
    # lag_168h cannot be constructed.
    payload["demand_history"] = (
        payload[
            "demand_history"
        ][-10:]
    )

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload
        )

        assert (
            response.status_code
            == 400
        )


# =========================================================
# TEST 5: MULTI-ZONE FLEET PREDICTION
# =========================================================

def test_fleet_prediction_endpoint():

    payload = (
        create_fleet_payload()
    )

    with TestClient(app) as client:

        response = client.post(
            "/predict/fleet",
            json=payload
        )

        assert (
            response.status_code
            == 200
        )

        result = (
            response.json()
        )

        assert (
            result[
                "zones_predicted"
            ]
            == 2
        )

        assert (
            result[
                "available_vehicles"
            ]
            == 100
        )

        recommendations = (
            result[
                "recommendations"
            ]
        )

        assert (
            len(recommendations)
            == 2
        )

        # All 100 vehicles must be allocated
        allocated = sum(
            item[
                "recommended_vehicles"
            ]
            for item
            in recommendations
        )

        assert allocated == 100

        # No negative predictions
        assert all(
            item[
                "predicted_demand"
            ] >= 0
            for item
            in recommendations
        )

        # Demand shares should sum to 1
        total_share = sum(
            item[
                "demand_share"
            ]
            for item
            in recommendations
        )

        assert abs(
            total_share - 1.0
        ) < 1e-9


# =========================================================
# TEST 6: INVALID FLEET SIZE
# =========================================================

def test_invalid_fleet_size():

    payload = (
        create_fleet_payload()
    )

    payload[
        "available_vehicles"
    ] = 0

    with TestClient(app) as client:

        response = client.post(
            "/predict/fleet",
            json=payload
        )

        # Pydantic gt=0 validation
        assert (
            response.status_code
            == 422
        )


# =========================================================
# TEST 7: EMPTY ZONE LIST
# =========================================================

def test_empty_zone_list():

    payload = (
        create_fleet_payload()
    )

    payload[
        "zone_ids"
    ] = []

    with TestClient(app) as client:

        response = client.post(
            "/predict/fleet",
            json=payload
        )

        assert (
            response.status_code
            == 400
        )