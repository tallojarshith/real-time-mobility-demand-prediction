from datetime import datetime

from src.database.repository import get_zone_demand_history
from src.ingestion.weather_service import WeatherService
from src.prediction.predictor import DemandPredictor

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ZONE_ID = 161

TARGET_TIMESTAMP = datetime(
    2025,
    2,
    1,
    0,
    0,
)


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("DATABASE + WEATHER + MODEL INTEGRATION TEST")
    print("=" * 60)

    print(
        f"\nZone: {ZONE_ID}"
    )

    print(
        f"Target timestamp: {TARGET_TIMESTAMP}"
    )

    # =====================================================
    # STEP 1 — LOAD DEMAND FROM POSTGRESQL
    # =====================================================

    print(
        "\n[1/4] Loading demand history from PostgreSQL..."
    )

    demand_history = (
        get_zone_demand_history(
            zone_id=ZONE_ID,
            target_timestamp=TARGET_TIMESTAMP,
            history_hours=168,
        )
    )

    print(
        f"Demand rows retrieved: "
        f"{len(demand_history)}"
    )

    print(
        "Demand history range:",
        demand_history[
            "pickup_hour_timestamp"
        ].min(),
        "->",
        demand_history[
            "pickup_hour_timestamp"
        ].max(),
    )

    # =====================================================
    # STEP 2 — GET TIME-ALIGNED WEATHER
    # =====================================================

    print(
        "\n[2/4] Fetching target-hour weather..."
    )

    weather_service = WeatherService()

    weather = (
        weather_service.get_weather_for_hour(
            TARGET_TIMESTAMP
        )
    )

    print(
        "Weather retrieved:"
    )

    for feature, value in weather.items():

        print(
            f"  {feature:25s}: {value}"
        )

    weather_row = pd.DataFrame(
        [weather]
    )

    # =====================================================
    # STEP 3 — LOAD MODEL
    # =====================================================

    print(
        "\n[3/4] Loading trained model..."
    )

    predictor = DemandPredictor()

    print(
        "Model loaded successfully."
    )

    print(
        "Feature count:",
        len(
            predictor.feature_columns
        ),
    )

    # =====================================================
    # STEP 4 — PREDICT
    # =====================================================

    print(
        "\n[4/4] Generating prediction..."
    )

    prediction = predictor.predict(
        demand_history=demand_history,
        weather_row=weather_row,
        target_timestamp=TARGET_TIMESTAMP,
        zone_id=ZONE_ID,
    )

    # =====================================================
    # RESULT
    # =====================================================

    print()
    print("=" * 60)
    print("FULL BACKEND INTEGRATION TEST PASSED")
    print("=" * 60)

    print(
        f"Zone ID: {ZONE_ID}"
    )

    print(
        f"Target timestamp: "
        f"{TARGET_TIMESTAMP}"
    )

    print(
        f"Predicted demand: "
        f"{prediction:.2f}"
    )

    print()
    print(
        "Pipeline verified:"
    )

    print(
        "PostgreSQL"
        " -> demand history"
        " -> historical weather"
        " -> feature engineering"
        " -> Extra Trees"
        " -> prediction"
    )


if __name__ == "__main__":
    main()