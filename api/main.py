from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prediction.predictor import DemandPredictor
from src.fleet.recommendations import create_fleet_recommendations
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
    version="1.0.0",
    lifespan=lifespan
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
        le=265
    )

    demand: float = Field(
        ge=0
    )


class WeatherInput(BaseModel):
    temperature_2m: float

    relative_humidity_2m: float

    precipitation: float = Field(
        ge=0
    )

    rain: float = Field(
        ge=0
    )

    snowfall: float = Field(
        ge=0
    )

    weather_code: int

    wind_speed_10m: float = Field(
        ge=0
    )


# =========================================================
# SINGLE-ZONE REQUEST
# =========================================================

class PredictionRequest(BaseModel):
    target_timestamp: datetime

    zone_id: int = Field(
        ge=1,
        le=265
    )

    demand_history: List[
        DemandHistoryRecord
    ]

    weather: WeatherInput


class PredictionResponse(BaseModel):
    target_timestamp: datetime
    zone_id: int
    predicted_demand: float


# =========================================================
# MULTI-ZONE / FLEET REQUEST
# =========================================================

class FleetPredictionRequest(BaseModel):
    target_timestamp: datetime

    available_vehicles: int = Field(
        gt=0
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
            "1.0.0",

        "docs":
            "/docs"
    }


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get(
    "/health",
    response_model=HealthResponse
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

        feature_count=feature_count
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
            predictor.feature_columns
    }


# =========================================================
# SINGLE-ZONE PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_demand(
    request: PredictionRequest
):
    """
    Predict next-hour pickup demand for one NYC taxi zone.

    Historical demand must contain sufficient continuous
    history for lag and rolling features.
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
            )
        )

    logger.info(
        "API prediction request | "
        "zone=%d | timestamp=%s",
        request.zone_id,
        request.target_timestamp
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
            zone_id=request.zone_id
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return PredictionResponse(
            target_timestamp=(
                request.target_timestamp
            ),
            zone_id=request.zone_id,
            predicted_demand=prediction
        )

    except ValueError as error:

        logger.warning(
            "Prediction request rejected | "
            "reason=%s",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Unexpected prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )


# =========================================================
# MULTI-ZONE + FLEET ENDPOINT
# =========================================================

@app.post(
    "/predict/fleet",
    response_model=FleetPredictionResponse
)
def predict_fleet(
    request: FleetPredictionRequest
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
            )
        )

    logger.info(
        "Fleet prediction request | "
        "zones=%d | vehicles=%d | timestamp=%s",
        len(request.zone_ids),
        request.available_vehicles,
        request.target_timestamp
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

        # Remove duplicate zone IDs
        # while preserving order.
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
                zone_ids=zone_ids
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
                )
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
                )
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
                allocated_vehicles
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
            allocated_vehicles
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

            recommendations=records
        )

    except ValueError as error:

        logger.warning(
            "Fleet prediction rejected | "
            "reason=%s",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Unexpected fleet prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal fleet prediction error"
            )
        )