from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prediction.predictor import DemandPredictor
from src.fleet.recommendations import create_fleet_recommendations
from src.ingestion.weather_service import WeatherService
from src.database.repository import get_zone_demand_history
from src.database.prediction_repository import save_prediction
from src.utils.logger import get_logger


# =========================================================
# LOGGER
# =========================================================

logger = get_logger(__name__)


# =========================================================
# APPLICATION STATE
# =========================================================

app_state = {}


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model once when FastAPI starts.

    This prevents the Extra Trees model from being loaded
    again for every prediction request.
    """

    logger.info(
        "Starting NYC Taxi Demand API"
    )

    predictor = DemandPredictor()

    app_state["predictor"] = predictor

    logger.info(
        "Prediction service loaded successfully"
    )

    yield

    logger.info(
        "Shutting down NYC Taxi Demand API"
    )

    app_state.clear()


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="NYC Taxi Demand Forecasting API",
    description=(
        "Next-hour NYC taxi demand forecasting "
        "and fleet decision-support API."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_count: int


class DemandHistoryRecord(BaseModel):
    pickup_hour_timestamp: datetime

    PULocationID: int = Field(
        ge=1,
        le=265,
    )

    demand: float = Field(
        ge=0,
    )


class WeatherInput(BaseModel):
    temperature_2m: float

    relative_humidity_2m: float

    precipitation: float = Field(
        ge=0,
    )

    rain: float = Field(
        ge=0,
    )

    snowfall: float = Field(
        ge=0,
    )

    weather_code: int

    wind_speed_10m: float = Field(
        ge=0,
    )


# =========================================================
# SINGLE-ZONE REQUEST
# =========================================================

class PredictionRequest(BaseModel):
    """
    Standard prediction request.

    Weather is supplied explicitly by the client.
    """

    target_timestamp: datetime

    zone_id: int = Field(
        ge=1,
        le=265,
    )

    demand_history: List[
        DemandHistoryRecord
    ]

    weather: WeatherInput


class AutoWeatherPredictionRequest(BaseModel):
    """
    Prediction request where target-hour weather
    is fetched automatically from Open-Meteo.
    """

    target_timestamp: datetime

    zone_id: int = Field(
        ge=1,
        le=265,
    )

    demand_history: List[
        DemandHistoryRecord
    ]


class PredictionResponse(BaseModel):
    target_timestamp: datetime
    zone_id: int
    predicted_demand: float


# =========================================================
# DATABASE-BACKED PREDICTION REQUEST / RESPONSE
# =========================================================

class DatabasePredictionRequest(BaseModel):
    target_timestamp: datetime

    zone_id: int = Field(
        ge=1,
        le=265,
    )


class DatabasePredictionResponse(BaseModel):
    prediction_id: int
    target_timestamp: datetime
    zone_id: int
    predicted_demand: float
    demand_history_rows: int
    weather_source: str


# =========================================================
# MULTI-ZONE / FLEET REQUEST
# =========================================================

class FleetPredictionRequest(BaseModel):
    target_timestamp: datetime

    available_vehicles: int = Field(
        gt=0,
    )

    zone_ids: List[int]

    demand_history: List[
        DemandHistoryRecord
    ]

    weather: WeatherInput


class FleetPredictionRecord(BaseModel):
    PULocationID: int
    predicted_demand: float
    recommended_vehicles: int
    demand_share: float
    priority: str


class FleetPredictionResponse(BaseModel):
    target_timestamp: datetime
    available_vehicles: int
    zones_predicted: int

    recommendations: List[
        FleetPredictionRecord
    ]


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "message":
            "NYC Taxi Demand Forecasting API",

        "version":
            "1.1.0",

        "docs":
            "/docs",
    }


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():
    """
    Check whether the API and model are available.
    """

    predictor = app_state.get(
        "predictor"
    )

    model_loaded = (
        predictor is not None
        and predictor.model is not None
    )

    feature_count = (
        len(predictor.feature_columns)
        if predictor is not None
        else 0
    )

    return HealthResponse(
        status=(
            "healthy"
            if model_loaded
            else "unhealthy"
        ),

        model_loaded=model_loaded,

        feature_count=feature_count,
    )


# =========================================================
# MODEL INFORMATION ENDPOINT
# =========================================================

@app.get("/model/info")
def model_info():
    """
    Return information about the deployed model.
    """

    predictor = app_state.get(
        "predictor"
    )

    if predictor is None:

        return {
            "model_loaded": False
        }

    return {
        "model_loaded": True,

        "model_type":
            type(predictor.model).__name__,

        "forecast_horizon":
            "1 hour",

        "evaluation_type":
            "rolling one-hour-ahead",

        "feature_count":
            len(
                predictor.feature_columns
            ),

        "features":
            predictor.feature_columns,
    }


# =========================================================
# SINGLE-ZONE PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_demand(
    request: PredictionRequest,
):
    """
    Predict next-hour pickup demand for one NYC taxi zone.

    Historical demand must contain sufficient continuous
    history for lag and rolling features.

    Weather is supplied explicitly by the API client.
    """

    predictor = app_state.get(
        "predictor"
    )

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service "
                "is not available"
            ),
        )

    logger.info(
        "API prediction request | "
        "zone=%d | timestamp=%s",
        request.zone_id,
        request.target_timestamp,
    )

    try:

        # -------------------------------------------------
        # Convert demand history to DataFrame
        # -------------------------------------------------

        demand_history = pd.DataFrame(
            [
                record.model_dump()
                for record
                in request.demand_history
            ]
        )

        # -------------------------------------------------
        # Convert weather to DataFrame
        # -------------------------------------------------

        weather_row = pd.DataFrame(
            [
                request.weather.model_dump()
            ]
        )

        # -------------------------------------------------
        # Generate prediction
        # -------------------------------------------------

        prediction = predictor.predict(
            demand_history=demand_history,
            weather_row=weather_row,
            target_timestamp=(
                request.target_timestamp
            ),
            zone_id=request.zone_id,
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return PredictionResponse(
            target_timestamp=(
                request.target_timestamp
            ),
            zone_id=request.zone_id,
            predicted_demand=prediction,
        )

    except ValueError as error:

        logger.warning(
            "Prediction request rejected | "
            "reason=%s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:

        logger.exception(
            "Unexpected prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error",
        )


# =========================================================
# SINGLE-ZONE PREDICTION WITH AUTOMATIC WEATHER
# =========================================================

@app.post(
    "/predict/auto-weather",
    response_model=PredictionResponse,
)
def predict_demand_auto_weather(
    request: AutoWeatherPredictionRequest,
):
    """
    Predict next-hour pickup demand for one NYC taxi zone.

    Weather features are fetched automatically from
    Open-Meteo for the requested target timestamp.

    Historical demand must still be supplied because
    lag and rolling features depend on recent observed
    taxi demand.
    """

    predictor = app_state.get(
        "predictor"
    )

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service "
                "is not available"
            ),
        )

    logger.info(
        "Auto-weather prediction request | "
        "zone=%d | timestamp=%s",
        request.zone_id,
        request.target_timestamp,
    )

    try:

        # -------------------------------------------------
        # Convert demand history to DataFrame
        # -------------------------------------------------

        demand_history = pd.DataFrame(
            [
                record.model_dump()
                for record
                in request.demand_history
            ]
        )

        # -------------------------------------------------
        # Fetch target-hour weather automatically
        # -------------------------------------------------

        weather_service = WeatherService()

        weather = (
            weather_service.get_weather_for_hour(
                request.target_timestamp
            )
        )

        logger.info(
            "Weather forecast retrieved | "
            "timestamp=%s",
            request.target_timestamp,
        )

        # -------------------------------------------------
        # Convert weather to model-ready DataFrame
        # -------------------------------------------------

        weather_row = pd.DataFrame(
            [weather]
        )

        # -------------------------------------------------
        # Generate prediction
        # -------------------------------------------------

        prediction = predictor.predict(
            demand_history=demand_history,
            weather_row=weather_row,
            target_timestamp=(
                request.target_timestamp
            ),
            zone_id=request.zone_id,
        )

        logger.info(
            "Auto-weather prediction completed | "
            "zone=%d | prediction=%.2f",
            request.zone_id,
            prediction,
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return PredictionResponse(
            target_timestamp=(
                request.target_timestamp
            ),
            zone_id=request.zone_id,
            predicted_demand=prediction,
        )

    except ValueError as error:

        logger.warning(
            "Auto-weather prediction rejected | "
            "reason=%s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except RuntimeError as error:

        # Open-Meteo/network/provider failure
        logger.error(
            "Weather service unavailable | "
            "reason=%s",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except Exception:

        logger.exception(
            "Unexpected auto-weather "
            "prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal prediction error"
            ),
        )


# =========================================================
# DATABASE-BACKED PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict/database",
    response_model=DatabasePredictionResponse,
)
def predict_from_database(
    request: DatabasePredictionRequest,
):
    """
    Production-style single-zone prediction.

    The client supplies only the taxi zone and target timestamp.
    Demand history is loaded from PostgreSQL, weather is fetched
    automatically, and the generated prediction is persisted.
    """

    predictor = app_state.get(
        "predictor"
    )

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service "
                "is not available"
            ),
        )

    logger.info(
        "Database prediction request | "
        "zone=%d | timestamp=%s",
        request.zone_id,
        request.target_timestamp,
    )

    try:
        demand_history = get_zone_demand_history(
            zone_id=request.zone_id,
            target_timestamp=request.target_timestamp,
            history_hours=168,
        )

        weather_service = WeatherService()
        weather = weather_service.get_weather_for_hour(
            request.target_timestamp
        )
        weather_row = pd.DataFrame([weather])

        current_hour = datetime.now().replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        target_hour = request.target_timestamp.replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        weather_source = (
            "open-meteo-archive"
            if target_hour < current_hour
            else "open-meteo-forecast"
        )

        prediction = predictor.predict(
            demand_history=demand_history,
            weather_row=weather_row,
            target_timestamp=request.target_timestamp,
            zone_id=request.zone_id,
        )

        prediction_id = save_prediction(
            target_timestamp=request.target_timestamp,
            zone_id=request.zone_id,
            predicted_demand=prediction,
            model_name=type(predictor.model).__name__,
        )

        logger.info(
            "Database prediction completed | "
            "id=%d | zone=%d | prediction=%.2f",
            prediction_id,
            request.zone_id,
            prediction,
        )

        return DatabasePredictionResponse(
            prediction_id=prediction_id,
            target_timestamp=request.target_timestamp,
            zone_id=request.zone_id,
            predicted_demand=prediction,
            demand_history_rows=len(demand_history),
            weather_source=weather_source,
        )

    except ValueError as error:
        logger.warning(
            "Database prediction rejected | reason=%s",
            error,
        )
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except RuntimeError as error:
        logger.error(
            "External service failure | reason=%s",
            error,
        )
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except Exception:
        logger.exception(
            "Unexpected database prediction error"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error",
        )


# =========================================================
# MULTI-ZONE + FLEET ENDPOINT
# =========================================================

@app.post(
    "/predict/fleet",
    response_model=FleetPredictionResponse,
)
def predict_fleet(
    request: FleetPredictionRequest,
):
    """
    Predict next-hour demand across multiple taxi zones
    and generate demand-proportional fleet recommendations.

    The fleet recommendation is a decision-support
    heuristic and not a routing optimization algorithm.
    """

    predictor = app_state.get(
        "predictor"
    )

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service "
                "is not available"
            ),
        )

    logger.info(
        "Fleet prediction request | "
        "zones=%d | vehicles=%d | timestamp=%s",
        len(request.zone_ids),
        request.available_vehicles,
        request.target_timestamp,
    )

    try:

        # -------------------------------------------------
        # Validate zone list
        # -------------------------------------------------

        if not request.zone_ids:

            raise ValueError(
                "At least one taxi zone is required"
            )

        invalid_zones = [
            zone_id
            for zone_id
            in request.zone_ids
            if zone_id < 1
            or zone_id > 265
        ]

        if invalid_zones:

            raise ValueError(
                f"Invalid taxi zones: "
                f"{invalid_zones}"
            )

        # Remove duplicate zone IDs while preserving order.
        zone_ids = list(
            dict.fromkeys(
                request.zone_ids
            )
        )

        # -------------------------------------------------
        # Convert demand history
        # -------------------------------------------------

        demand_history = pd.DataFrame(
            [
                record.model_dump()
                for record
                in request.demand_history
            ]
        )

        # -------------------------------------------------
        # Convert weather
        # -------------------------------------------------

        weather_row = pd.DataFrame(
            [
                request.weather.model_dump()
            ]
        )

        # -------------------------------------------------
        # Generate multi-zone predictions
        # -------------------------------------------------

        predictions = (
            predictor.predict_multiple_zones(
                demand_history=(
                    demand_history
                ),
                weather_row=(
                    weather_row
                ),
                target_timestamp=(
                    request.target_timestamp
                ),
                zone_ids=zone_ids,
            )
        )

        # -------------------------------------------------
        # Generate fleet recommendations
        # -------------------------------------------------

        recommendations = (
            create_fleet_recommendations(
                predictions=predictions,
                available_vehicles=(
                    request.available_vehicles
                ),
            )
        )

        # pd.cut creates a categorical column.
        # Convert it to normal strings for JSON.
        recommendations["priority"] = (
            recommendations[
                "priority"
            ].astype(str)
        )

        # -------------------------------------------------
        # Build response records
        # -------------------------------------------------

        records = []

        for _, row in (
            recommendations.iterrows()
        ):

            record = FleetPredictionRecord(
                PULocationID=int(
                    row["PULocationID"]
                ),

                predicted_demand=float(
                    row["predicted_demand"]
                ),

                recommended_vehicles=int(
                    row["recommended_vehicles"]
                ),

                demand_share=float(
                    row["demand_share"]
                ),

                priority=str(
                    row["priority"]
                ),
            )

            records.append(
                record
            )

        # -------------------------------------------------
        # Final validation
        # -------------------------------------------------

        allocated_vehicles = sum(
            record.recommended_vehicles
            for record in records
        )

        if (
            allocated_vehicles
            != request.available_vehicles
        ):

            logger.error(
                "Fleet allocation mismatch | "
                "available=%d | allocated=%d",
                request.available_vehicles,
                allocated_vehicles,
            )

            raise ValueError(
                "Fleet allocation does not match "
                "available vehicle count"
            )

        logger.info(
            "Fleet API request completed | "
            "predicted zones=%d | "
            "allocated vehicles=%d",
            len(records),
            allocated_vehicles,
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return FleetPredictionResponse(
            target_timestamp=(
                request.target_timestamp
            ),

            available_vehicles=(
                request.available_vehicles
            ),

            zones_predicted=len(
                records
            ),

            recommendations=records,
        )

    except ValueError as error:

        logger.warning(
            "Fleet prediction rejected | "
            "reason=%s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:

        logger.exception(
            "Unexpected fleet prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal fleet prediction error"
            ),
        )