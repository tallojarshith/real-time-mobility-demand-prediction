from pathlib import Path

import pandas as pd
import requests


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TAXI_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)

WEATHER_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "nyc_weather_2025_01.parquet"
)

API_URL = "http://127.0.0.1:8000/predict"

ZONE_ID = 161

TARGET_TIMESTAMP = pd.Timestamp(
    "2025-02-01 00:00:00"
)


# =========================================================
# LOAD TAXI DATA
# =========================================================

print("Loading January taxi data...")

taxi_df = pd.read_parquet(
    TAXI_DATA_PATH,
    columns=[
        "tpep_pickup_datetime",
        "PULocationID",
    ],
)

taxi_df["tpep_pickup_datetime"] = pd.to_datetime(
    taxi_df["tpep_pickup_datetime"]
)

taxi_df = taxi_df[
    (
        taxi_df["tpep_pickup_datetime"]
        >= "2025-01-01"
    )
    &
    (
        taxi_df["tpep_pickup_datetime"]
        < "2025-02-01"
    )
].copy()


# =========================================================
# CREATE HOURLY DEMAND FOR ZONE
# =========================================================

zone_df = taxi_df[
    taxi_df["PULocationID"] == ZONE_ID
].copy()

zone_df["pickup_hour_timestamp"] = (
    zone_df["tpep_pickup_datetime"]
    .dt.floor("h")
)

hourly_demand = (
    zone_df
    .groupby("pickup_hour_timestamp")
    .size()
    .rename("demand")
)


# =========================================================
# COMPLETE HOURLY GRID
# =========================================================

all_hours = pd.date_range(
    start="2025-01-01 00:00:00",
    end="2025-02-01 00:00:00",
    freq="h",
    inclusive="left",
)

history_df = (
    hourly_demand
    .reindex(
        all_hours,
        fill_value=0,
    )
    .rename_axis(
        "pickup_hour_timestamp"
    )
    .reset_index()
)

history_df["PULocationID"] = ZONE_ID


# =========================================================
# VALIDATE HISTORY
# =========================================================

if len(history_df) < 168:

    raise ValueError(
        "Insufficient historical demand "
        "for lag_168h"
    )


if history_df[
    "pickup_hour_timestamp"
].duplicated().any():

    raise ValueError(
        "Duplicate hourly timestamps "
        "found in demand history"
    )


print(
    f"Zone {ZONE_ID} history rows: "
    f"{len(history_df)}"
)

print(
    "History range:",
    history_df[
        "pickup_hour_timestamp"
    ].min(),
    "->",
    history_df[
        "pickup_hour_timestamp"
    ].max(),
)


# =========================================================
# LOAD WEATHER DATA
# =========================================================

print()
print("Loading weather data...")

weather_df = pd.read_parquet(
    WEATHER_DATA_PATH
)

print(
    "Weather columns:",
    list(weather_df.columns),
)


# =========================================================
# DETECT WEATHER TIMESTAMP COLUMN
# =========================================================

possible_timestamp_columns = [
    "time",
    "pickup_hour_timestamp",
    "timestamp",
    "datetime",
]


weather_time_column = None


for column in possible_timestamp_columns:

    if column in weather_df.columns:

        weather_time_column = column

        break


if weather_time_column is None:

    raise ValueError(
        "Could not find weather timestamp column. "
        f"Available columns: "
        f"{list(weather_df.columns)}"
    )


print(
    "Detected weather timestamp column:",
    weather_time_column,
)


weather_df[
    weather_time_column
] = pd.to_datetime(
    weather_df[
        weather_time_column
    ],
    errors="coerce",
)


weather_df = weather_df.dropna(
    subset=[
        weather_time_column
    ]
)


weather_df = weather_df.sort_values(
    weather_time_column
)


# =========================================================
# VALIDATE REQUIRED WEATHER FEATURES
# =========================================================

required_weather_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
]


missing_weather_features = [
    column
    for column
    in required_weather_features
    if column not in weather_df.columns
]


if missing_weather_features:

    raise ValueError(
        "Missing required weather features: "
        f"{missing_weather_features}"
    )


# =========================================================
# GET LATEST AVAILABLE WEATHER
# =========================================================

latest_weather = (
    weather_df
    .iloc[-1]
)


latest_weather_timestamp = (
    latest_weather[
        weather_time_column
    ]
)


print(
    "Latest available weather timestamp:",
    latest_weather_timestamp,
)


# IMPORTANT:
# January weather ends before our Feb 1 target.
#
# This latest weather row is used ONLY to verify that
# the Dockerized API inference pipeline works.
#
# A real production forecast must fetch weather for
# the actual target prediction hour.


weather_payload = {

    "temperature_2m":
        float(
            latest_weather[
                "temperature_2m"
            ]
        ),

    "relative_humidity_2m":
        float(
            latest_weather[
                "relative_humidity_2m"
            ]
        ),

    "precipitation":
        float(
            latest_weather[
                "precipitation"
            ]
        ),

    "rain":
        float(
            latest_weather[
                "rain"
            ]
        ),

    "snowfall":
        float(
            latest_weather[
                "snowfall"
            ]
        ),

    "weather_code":
        float(
            latest_weather[
                "weather_code"
            ]
        ),

    "wind_speed_10m":
        float(
            latest_weather[
                "wind_speed_10m"
            ]
        ),
}


# =========================================================
# BUILD HISTORY PAYLOAD
# =========================================================

history_payload = []


for row in history_df.itertuples(
    index=False
):

    history_payload.append({

        "pickup_hour_timestamp":
            row
            .pickup_hour_timestamp
            .isoformat(),

        "PULocationID":
            int(
                row.PULocationID
            ),

        "demand":
            int(
                row.demand
            ),
    })


# =========================================================
# BUILD API PAYLOAD
# =========================================================

payload = {

    "target_timestamp":
        TARGET_TIMESTAMP.isoformat(),

    "zone_id":
        ZONE_ID,

    "demand_history":
        history_payload,

    "weather":
        weather_payload,
}


# =========================================================
# CALL DOCKERIZED API
# =========================================================

print()
print(
    f"Sending prediction request to "
    f"{API_URL}"
)


response = requests.post(
    API_URL,
    json=payload,
    timeout=60,
)


# =========================================================
# DISPLAY RESPONSE
# =========================================================

print(
    "HTTP status:",
    response.status_code,
)


try:

    response_body = (
        response.json()
    )

except ValueError:

    print(
        "API returned a non-JSON response:"
    )

    print(
        response.text
    )

    raise


if response.status_code != 200:

    print()
    print(
        "Prediction request failed."
    )

    print(
        response_body
    )

    raise SystemExit(1)


# =========================================================
# SUCCESS
# =========================================================

print()
print(
    "======================================"
)

print(
    "DOCKER PREDICTION TEST PASSED"
)

print(
    "======================================"
)

print(
    f"Zone ID: {ZONE_ID}"
)

print(
    f"Target timestamp: "
    f"{TARGET_TIMESTAMP}"
)

print(
    f"History rows sent: "
    f"{len(history_payload)}"
)

print(
    f"Weather timestamp used: "
    f"{latest_weather_timestamp}"
)

print()
print(
    "API response:"
)

print(
    response_body
)