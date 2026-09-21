from datetime import datetime

from src.ingestion.weather_service import WeatherService


def main():

    print("=" * 50)
    print("NYC WEATHER SERVICE TEST")
    print("=" * 50)

    weather_service = WeatherService()

    # -----------------------------------------------------
    # Fetch the forecast first
    # -----------------------------------------------------

    print("\nFetching weather forecast from Open-Meteo...")

    forecast = weather_service.fetch_hourly_forecast()

    hourly = forecast["hourly"]

    print("Weather API connection successful.")

    print(
        "Forecast range:",
        hourly["time"][0],
        "->",
        hourly["time"][-1],
    )

    print(
        "Number of forecast hours:",
        len(hourly["time"]),
    )

    # -----------------------------------------------------
    # Pick an hour that definitely exists in the response
    # -----------------------------------------------------

    # We intentionally use one of the returned forecast
    # timestamps instead of hard-coding today's date.
    # This makes the test reusable in the future.

    test_timestamp_string = hourly["time"][1]

    test_timestamp = datetime.fromisoformat(
        test_timestamp_string
    )

    print(
        "\nTesting weather retrieval for:",
        test_timestamp,
    )

    # -----------------------------------------------------
    # Retrieve model-ready weather features
    # -----------------------------------------------------

    weather = weather_service.get_weather_for_hour(
        test_timestamp
    )

    print("\nWeather features:")

    for feature, value in weather.items():
        print(
            f"{feature:25s}: {value}"
        )

    # -----------------------------------------------------
    # Validate feature schema
    # -----------------------------------------------------

    expected_features = {
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "wind_speed_10m",
    }

    returned_features = set(
        weather.keys()
    )

    if returned_features != expected_features:

        raise ValueError(
            "Weather feature schema mismatch.\n"
            f"Expected: {expected_features}\n"
            f"Received: {returned_features}"
        )

    print()
    print("=" * 50)
    print("WEATHER SERVICE TEST PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()