import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def add_zone_metadata(
    predictions: pd.DataFrame,
    zone_lookup: pd.DataFrame
) -> pd.DataFrame:
    """
    Add taxi-zone name, borough and service-zone metadata
    to model predictions.
    """

    required_prediction_columns = [
        "PULocationID",
        "predicted_demand"
    ]

    required_lookup_columns = [
        "LocationID",
        "Borough",
        "Zone",
        "service_zone"
    ]

    missing_prediction_columns = [
        col for col in required_prediction_columns
        if col not in predictions.columns
    ]

    missing_lookup_columns = [
        col for col in required_lookup_columns
        if col not in zone_lookup.columns
    ]

    if missing_prediction_columns:
        raise ValueError(
            f"Missing prediction columns: "
            f"{missing_prediction_columns}"
        )

    if missing_lookup_columns:
        raise ValueError(
            f"Missing zone lookup columns: "
            f"{missing_lookup_columns}"
        )

    logger.info(
        "Adding taxi-zone metadata to %d predictions",
        len(predictions)
    )

    enriched = predictions.merge(
        zone_lookup[
            [
                "LocationID",
                "Borough",
                "Zone",
                "service_zone"
            ]
        ],
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    )

    enriched = enriched.drop(
        columns=["LocationID"]
    )

    # TLC lookup itself contains some missing labels,
    # so preserve the zone ID and use display-safe labels.
    enriched["Zone"] = enriched["Zone"].fillna(
        "Unknown Zone"
    )

    enriched["Borough"] = enriched["Borough"].fillna(
        "Unknown"
    )

    enriched["service_zone"] = (
        enriched["service_zone"]
        .fillna("Unknown")
    )

    logger.info(
        "Taxi-zone metadata added successfully"
    )

    return enriched


def create_fleet_recommendations(
    predictions: pd.DataFrame,
    available_vehicles: int
) -> pd.DataFrame:
    """
    Create demand-proportional fleet allocation recommendations.

    IMPORTANT:
    This is a decision-support heuristic, not a mathematical
    fleet-routing optimizer.

    Vehicles are allocated approximately in proportion to
    predicted pickup demand.
    """

    if available_vehicles <= 0:
        raise ValueError(
            "available_vehicles must be greater than 0"
        )

    if "predicted_demand" not in predictions.columns:
        raise ValueError(
            "predicted_demand column is required"
        )

    result = predictions.copy()

    result["predicted_demand"] = (
        pd.to_numeric(
            result["predicted_demand"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    total_demand = result[
        "predicted_demand"
    ].sum()

    if total_demand <= 0:
        raise ValueError(
            "Total predicted demand must be greater than 0"
        )

    logger.info(
        "Creating fleet recommendations | "
        "vehicles=%d | total predicted demand=%.2f",
        available_vehicles,
        total_demand
    )

    # Demand share for each zone
    result["demand_share"] = (
        result["predicted_demand"]
        / total_demand
    )

    # Initial proportional allocation
    exact_allocation = (
        result["demand_share"]
        * available_vehicles
    )

    result["recommended_vehicles"] = (
        exact_allocation
        .astype(int)
    )

    # -----------------------------------------------------
    # Distribute remaining vehicles using largest remainder
    # -----------------------------------------------------

    remaining = (
        available_vehicles
        - result["recommended_vehicles"].sum()
    )

    remainders = (
        exact_allocation
        - result["recommended_vehicles"]
    )

    if remaining > 0:

        priority_indices = (
            remainders
            .sort_values(ascending=False)
            .index[:remaining]
        )

        result.loc[
            priority_indices,
            "recommended_vehicles"
        ] += 1

    # Operational priority based on relative demand
    result["priority"] = pd.cut(
        result["demand_share"],
        bins=[
            -float("inf"),
            0.01,
            0.03,
            float("inf")
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    result = result.sort_values(
        [
            "recommended_vehicles",
            "predicted_demand"
        ],
        ascending=False
    ).reset_index(drop=True)

    logger.info(
        "Fleet recommendations created successfully | "
        "allocated vehicles=%d",
        result["recommended_vehicles"].sum()
    )

    return result