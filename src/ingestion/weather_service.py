from __future__ import annotations

from datetime import datetime

import requests


# =========================================================
# NYC LOCATION
# =========================================================

NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060


# =========================================================
# OPEN-METEO ENDPOINTS
# =========================================================

FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# =========================================================
# WEATHER SERVICE
# =========================================================

class WeatherService:
    """
    Retrieve hourly NYC weather features required by
    the taxi demand forecasting model.

    The service supports:

    1. Historical weather through Open-Meteo Archive API.
    2. Current/future weather through Open-Meteo Forecast API.

    The correct source is selected automatically based
    on the requested target timestamp.
    """

    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "wind_speed_10m",
    ]

    def __init__(
        self,
        latitude: float = NYC_LATITUDE,
        longitude: float = NYC_LONGITUDE,
    ):
        self.latitude = latitude
        self.longitude = longitude

    # =====================================================
    # HTTP REQUEST
    # =====================================================

    def _request(
        self,
        url: str,
        params: dict,
    ) -> dict:
        """
        Execute an Open-Meteo API request.
        """

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            raise RuntimeError(
                "Failed to fetch weather data "
                f"from Open-Meteo: {exc}"
            ) from exc

        data = response.json()

        if "hourly" not in data:

            raise ValueError(
                "Open-Meteo response does not "
                "contain hourly weather data."
            )

        return data

    # =====================================================
    # FORECAST WEATHER
    # =====================================================

    def fetch_hourly_forecast(
        self,
    ) -> dict:
        """
        Fetch the current seven-day hourly NYC forecast.
        """

        params = {
            "latitude":
                self.latitude,

            "longitude":
                self.longitude,

            "hourly":
                ",".join(
                    self.HOURLY_VARIABLES
                ),

            "timezone":
                "America/New_York",

            "forecast_days":
                7,
        }

        return self._request(
            FORECAST_URL,
            params,
        )

    # =====================================================
    # HISTORICAL WEATHER
    # =====================================================

    def fetch_historical_weather(
        self,
        target_timestamp: datetime,
    ) -> dict:
        """
        Fetch historical hourly weather for the date
        containing target_timestamp.
        """

        target_date = (
            target_timestamp
            .date()
            .isoformat()
        )

        params = {
            "latitude":
                self.latitude,

            "longitude":
                self.longitude,

            "start_date":
                target_date,

            "end_date":
                target_date,

            "hourly":
                ",".join(
                    self.HOURLY_VARIABLES
                ),

            "timezone":
                "America/New_York",
        }

        return self._request(
            ARCHIVE_URL,
            params,
        )

    # =====================================================
    # EXTRACT TARGET HOUR
    # =====================================================

    def _extract_weather_for_hour(
        self,
        data: dict,
        target_timestamp: datetime,
    ) -> dict:
        """
        Extract the seven model weather features for
        the requested target hour.
        """

        hourly = data["hourly"]

        if "time" not in hourly:

            raise ValueError(
                "Weather response does not "
                "contain hourly timestamps."
            )

        target_hour = (
            target_timestamp.replace(
                minute=0,
                second=0,
                microsecond=0,
            )
        )

        target_string = (
            target_hour.strftime(
                "%Y-%m-%dT%H:%M"
            )
        )

        try:

            index = (
                hourly["time"]
                .index(
                    target_string
                )
            )

        except ValueError as exc:

            raise ValueError(
                "No weather data available for "
                f"{target_string}."
            ) from exc

        weather = {}

        for feature in (
            self.HOURLY_VARIABLES
        ):

            if feature not in hourly:

                raise ValueError(
                    f"Weather feature "
                    f"'{feature}' missing "
                    "from Open-Meteo response."
                )

            value = (
                hourly[feature][index]
            )

            if value is None:

                raise ValueError(
                    f"Weather feature "
                    f"'{feature}' is missing "
                    f"for {target_string}."
                )

            weather[feature] = (
                float(value)
            )

        return weather

    # =====================================================
    # MAIN WEATHER METHOD
    # =====================================================

    def get_weather_for_hour(
        self,
        target_timestamp: datetime,
    ) -> dict:
        """
        Retrieve model-ready weather for a target hour.

        Historical timestamps use the Open-Meteo Archive
        API. Current/future timestamps use the Forecast API.
        """

        # -------------------------------------------------
        # Normalize target
        # -------------------------------------------------

        target_hour = (
            target_timestamp.replace(
                minute=0,
                second=0,
                microsecond=0,
            )
        )

        now = datetime.now()

        current_hour = (
            now.replace(
                minute=0,
                second=0,
                microsecond=0,
            )
        )

        # -------------------------------------------------
        # Historical weather
        # -------------------------------------------------

        if target_hour < current_hour:

            data = (
                self.fetch_historical_weather(
                    target_hour
                )
            )

        # -------------------------------------------------
        # Current / forecast weather
        # -------------------------------------------------

        else:

            data = (
                self.fetch_hourly_forecast()
            )

        # -------------------------------------------------
        # Extract model features
        # -------------------------------------------------

        return (
            self._extract_weather_for_hour(
                data=data,
                target_timestamp=target_hour,
            )
        )