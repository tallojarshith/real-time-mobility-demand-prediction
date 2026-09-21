from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from src.database.config import SessionLocal
from src.database.models import (
    HourlyDemand,
    IngestionMetadata,
)
from src.utils.logger import get_logger


# =========================================================
# CONFIGURATION
# =========================================================

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TAXI_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)

YEAR = 2025
MONTH = 1
DATASET = "yellow_taxi"


# =========================================================
# LOAD AND CLEAN RAW DATA
# =========================================================

def load_january_data() -> pd.DataFrame:

    logger.info(
        "Loading taxi data from %s",
        TAXI_DATA_PATH,
    )

    df = pd.read_parquet(
        TAXI_DATA_PATH,
        columns=[
            "tpep_pickup_datetime",
            "PULocationID",
        ],
    )

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    # Keep January 2025 only.
    df = df[
        (
            df["tpep_pickup_datetime"]
            >= "2025-01-01"
        )
        &
        (
            df["tpep_pickup_datetime"]
            < "2025-02-01"
        )
        &
        (
            df["PULocationID"]
            .between(1, 265)
        )
    ].copy()

    logger.info(
        "Clean January trips: %d",
        len(df),
    )

    return df


# =========================================================
# BUILD COMPLETE ZONE-HOUR DEMAND GRID
# =========================================================

def create_hourly_demand(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["pickup_hour_timestamp"] = (
        df["tpep_pickup_datetime"]
        .dt.floor("h")
    )

    observed = (
        df.groupby(
            [
                "pickup_hour_timestamp",
                "PULocationID",
            ]
        )
        .size()
        .rename("demand")
        .reset_index()
    )

    # Use zones actually present in this dataset.
    active_zones = sorted(
        df["PULocationID"]
        .dropna()
        .astype(int)
        .unique()
    )

    hours = pd.date_range(
        start="2025-01-01 00:00:00",
        end="2025-02-01 00:00:00",
        freq="h",
        inclusive="left",
    )

    complete_index = pd.MultiIndex.from_product(
        [
            hours,
            active_zones,
        ],
        names=[
            "pickup_hour_timestamp",
            "PULocationID",
        ],
    )

    hourly = (
        observed
        .set_index(
            [
                "pickup_hour_timestamp",
                "PULocationID",
            ]
        )
        .reindex(
            complete_index,
            fill_value=0,
        )
        .reset_index()
    )

    hourly["PULocationID"] = (
        hourly["PULocationID"]
        .astype(int)
    )

    hourly["demand"] = (
        hourly["demand"]
        .astype(float)
    )

    logger.info(
        "Hourly demand rows created: %d",
        len(hourly),
    )

    logger.info(
        "Active zones: %d",
        len(active_zones),
    )

    logger.info(
        "Total demand after aggregation: %.0f",
        hourly["demand"].sum(),
    )

    return hourly


# =========================================================
# WRITE TO POSTGRESQL
# =========================================================

def save_hourly_demand(
    hourly: pd.DataFrame,
) -> None:

    session = SessionLocal()

    try:

        records = [
            {
                "pickup_hour_timestamp":
                    row.pickup_hour_timestamp,

                "zone_id":
                    int(row.PULocationID),

                "demand":
                    float(row.demand),
            }

            for row in hourly.itertuples(
                index=False
            )
        ]

        batch_size = 5000

        inserted_rows = 0

        for start in range(
            0,
            len(records),
            batch_size,
        ):

            batch = records[
                start:start + batch_size
            ]

            statement = (
                insert(HourlyDemand)
                .values(batch)
                .on_conflict_do_nothing(
                    index_elements=[
                        "pickup_hour_timestamp",
                        "zone_id",
                    ]
                )
            )

            result = session.execute(
                statement
            )

            if result.rowcount:
                inserted_rows += (
                    result.rowcount
                )

        # ---------------------------------------------
        # Ingestion metadata
        # ---------------------------------------------

        metadata_statement = (
            insert(IngestionMetadata)
            .values(
                dataset=DATASET,
                year=YEAR,
                month=MONTH,
                rows_processed=len(hourly),
                status="success",
            )
            .on_conflict_do_update(
                index_elements=[
                    "dataset",
                    "year",
                    "month",
                ],
                set_={
                    "rows_processed":
                        len(hourly),

                    "status":
                        "success",
                },
            )
        )

        session.execute(
            metadata_statement
        )

        session.commit()

        logger.info(
            "Rows inserted into PostgreSQL: %d",
            inserted_rows,
        )

    except Exception:

        session.rollback()

        logger.exception(
            "Database ingestion failed"
        )

        raise

    finally:

        session.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("JANUARY 2025 TAXI DEMAND → POSTGRESQL")
    print("=" * 60)

    taxi_df = load_january_data()

    hourly = create_hourly_demand(
        taxi_df
    )

    # Important sanity check from our original pipeline.
    raw_total = len(
        taxi_df
    )

    aggregated_total = int(
        hourly["demand"].sum()
    )

    if raw_total != aggregated_total:

        raise ValueError(
            "Demand aggregation sanity check failed: "
            f"trips={raw_total}, "
            f"aggregated={aggregated_total}"
        )

    print(
        "\nAggregation sanity check passed:"
    )

    print(
        f"Trips: {raw_total:,}"
    )

    print(
        f"Aggregated demand: "
        f"{aggregated_total:,}"
    )

    print(
        f"Hourly rows: "
        f"{len(hourly):,}"
    )

    print(
        "\nWriting to PostgreSQL..."
    )

    save_hourly_demand(
        hourly
    )

    print()
    print("=" * 60)
    print("POSTGRESQL INGESTION PASSED")
    print("=" * 60)

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Period: {YEAR}-{MONTH:02d}"
    )

    print(
        f"Rows processed: "
        f"{len(hourly):,}"
    )


if __name__ == "__main__":
    main()