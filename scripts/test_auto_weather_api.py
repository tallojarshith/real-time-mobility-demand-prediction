from datetime import datetime, timedelta

import requests


API_URL = "http://127.0.0.1:8000/predict/auto-weather"

ZONE_ID = 161


def main():

    # -----------------------------------------------------
    # Determine a forecastable target hour
    # -----------------------------------------------------

    now = datetime.now()

    target_timestamp = (
        now.replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        + timedelta(hours=1)
    )

    print("=" * 55)
    print("AUTO-WEATHER API INTEGRATION TEST")
    print("=" * 55)

    print(
        "\nTarget timestamp:",
        target_timestamp,
    )

    # -----------------------------------------------------
    # Create synthetic recent demand history
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # This is ONLY an API integration test.
    #
    # We are NOT claiming this synthetic history represents
    # real current NYC taxi demand.
    #
    # Production inference requires recent observed demand
    # from the actual demand ingestion/storage system.
    # -----------------------------------------------------

    history = []

    history_start = (
        target_timestamp
        - timedelta(hours=168)
    )

    for hour_number in range(168):

        timestamp = (
            history_start
            + timedelta(hours=hour_number)
        )

        # Simple deterministic test pattern.
        # We deliberately avoid random values so that
        # the integration test is reproducible.

        hour = timestamp.hour

        if 7 <= hour <= 9:
            demand = 180

        elif 16 <= hour <= 20:
            demand = 220

        elif 0 <= hour <= 5:
            demand = 50

        else:
            demand = 120

        history.append(
            {
                "pickup_hour_timestamp":
                    timestamp.isoformat(),

                "PULocationID":
                    ZONE_ID,

                "demand":
                    demand,
            }
        )

    # -----------------------------------------------------
    # Build request
    # -----------------------------------------------------

    payload = {
        "target_timestamp":
            target_timestamp.isoformat(),

        "zone_id":
            ZONE_ID,

        "demand_history":
            history,
    }

    print(
        "History rows:",
        len(history),
    )

    print(
        "History range:",
        history[0][
            "pickup_hour_timestamp"
        ],
        "->",
        history[-1][
            "pickup_hour_timestamp"
        ],
    )

    print(
        "\nNotice: NO WEATHER is being "
        "sent by this client."
    )

    # -----------------------------------------------------
    # Call API
    # -----------------------------------------------------

    print(
        "\nCalling:",
        API_URL,
    )

    response = requests.post(
        API_URL,
        json=payload,
        timeout=60,
    )

    print(
        "HTTP status:",
        response.status_code,
    )

    try:

        body = response.json()

    except ValueError:

        print(
            "Non-JSON response:"
        )

        print(
            response.text
        )

        raise

    if response.status_code != 200:

        print(
            "\nRequest failed:"
        )

        print(
            body
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Success
    # -----------------------------------------------------

    print()
    print("=" * 55)
    print("AUTO-WEATHER API TEST PASSED")
    print("=" * 55)

    print(
        "Zone:",
        body["zone_id"],
    )

    print(
        "Target:",
        body["target_timestamp"],
    )

    print(
        "Predicted demand:",
        body["predicted_demand"],
    )

    print()
    print(
        "FastAPI successfully:"
    )

    print(
        "1. Received demand history"
    )

    print(
        "2. Fetched target-hour weather "
        "from Open-Meteo"
    )

    print(
        "3. Built the model features"
    )

    print(
        "4. Ran Extra Trees inference"
    )

    print(
        "5. Returned the prediction"
    )

    print()
    print(
        "NOTE: Synthetic demand was used "
        "only for integration testing."
    )


if __name__ == "__main__":
    main()