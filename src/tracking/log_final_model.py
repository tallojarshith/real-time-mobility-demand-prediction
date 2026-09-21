from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "final_extra_trees_model.joblib"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "feature_columns.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "model_metadata.json"
)

MLFLOW_DB_PATH = (
    PROJECT_ROOT
    / "mlflow.db"
)


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

EXPERIMENT_NAME = (
    "nyc-taxi-demand-forecasting"
)

RUN_NAME = (
    "final-tuned-extra-trees"
)


FINAL_METRICS = {
    "test_mae": 3.220,
    "test_rmse": 10.004,
    "test_r2": 0.974,
}


# =========================================================
# VALIDATE ARTIFACTS
# =========================================================

def validate_artifacts():

    required_files = [
        MODEL_PATH,
        FEATURE_PATH,
        METADATA_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Missing required artifacts:\n"
            + "\n".join(missing_files)
        )


# =========================================================
# BUILD SQLITE TRACKING URI
# =========================================================

def get_tracking_uri():

    # Convert Windows backslashes to forward slashes.
    database_path = (
        MLFLOW_DB_PATH
        .resolve()
        .as_posix()
    )

    return (
        f"sqlite:///{database_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # VALIDATE FILES
    # -----------------------------------------------------

    validate_artifacts()

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    print(
        f"Loaded model: "
        f"{type(model).__name__}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    # -----------------------------------------------------
    # CONFIGURE MLFLOW
    # -----------------------------------------------------

    tracking_uri = (
        get_tracking_uri()
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    print(
        f"MLflow tracking URI: "
        f"{tracking_uri}"
    )

    # -----------------------------------------------------
    # CREATE / SELECT EXPERIMENT
    # -----------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    # -----------------------------------------------------
    # START RUN
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name=RUN_NAME
    ) as run:

        # =================================================
        # PARAMETERS
        # =================================================

        parameters = {

            "model_type":
                type(model).__name__,

            "n_estimators":
                model.n_estimators,

            "max_depth":
                model.max_depth,

            "min_samples_split":
                model.min_samples_split,

            "min_samples_leaf":
                model.min_samples_leaf,

            "max_features":
                model.max_features,

            "random_state":
                model.random_state,

            "feature_count":
                len(feature_columns),

            "forecast_horizon":
                "1_hour",

            "evaluation_type":
                "rolling_one_hour_ahead",

            "training_data":
                "NYC_TLC_Yellow_Taxi_January_2025",
        }

        mlflow.log_params(
            parameters
        )

        # =================================================
        # TEST METRICS
        # =================================================

        mlflow.log_metrics(
            FINAL_METRICS
        )

        # =================================================
        # TAGS
        # =================================================

        mlflow.set_tags({

            "project":
                "NYC Taxi Demand Forecasting",

            "data_source":
                "NYC TLC Yellow Taxi",

            "data_period":
                "January 2025",

            "model_stage":
                "final",

            "task":
                "zone_level_demand_forecasting",

            "framework":
                "scikit-learn",
        })

        # =================================================
        # FEATURE SCHEMA
        # =================================================

        feature_text = "\n".join(
            str(feature)
            for feature
            in feature_columns
        )

        mlflow.log_text(
            feature_text,
            "metadata/feature_columns.txt",
        )

        # =================================================
        # METADATA JSON
        # =================================================

        mlflow.log_dict(
            metadata,
            "metadata/model_metadata.json",
        )

        # =================================================
        # MODEL SUMMARY
        # =================================================

        summary = f"""
NYC Taxi Demand Forecasting
===========================

Model
-----
{type(model).__name__}

Forecast Horizon
----------------
1 hour

Evaluation Strategy
-------------------
Rolling one-hour-ahead

Feature Count
-------------
{len(feature_columns)}

Final Held-Out Test Results
---------------------------
MAE  : {FINAL_METRICS["test_mae"]}
RMSE : {FINAL_METRICS["test_rmse"]}
R2   : {FINAL_METRICS["test_r2"]}

Model Configuration
-------------------
n_estimators       : {model.n_estimators}
max_depth          : {model.max_depth}
min_samples_split  : {model.min_samples_split}
min_samples_leaf   : {model.min_samples_leaf}
max_features       : {model.max_features}

Methodological Notes
--------------------
Historical demand lag and rolling features use only
information available before the target prediction hour.

The final evaluation uses rolling one-hour-ahead
forecasting rather than a four-day recursive forecast.

Weather features were retained in the experimental
pipeline, although their incremental validation
improvement during January 2025 was small.
"""

        mlflow.log_text(
            summary.strip(),
            "metadata/model_summary.txt",
        )

        # =================================================
        # LOG SKLEARN MODEL
        # =================================================

        mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ],
)

        # =================================================
        # SUCCESS OUTPUT
        # =================================================

        print()
        print(
            "========================================"
        )

        print(
            "MLflow logging completed successfully."
        )

        print(
            "========================================"
        )

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )

        print(
            f"Experiment ID: "
            f"{run.info.experiment_id}"
        )

        print(
            f"Model: "
            f"{type(model).__name__}"
        )

        print(
            f"Features: "
            f"{len(feature_columns)}"
        )

        print(
            f"Test MAE: "
            f"{FINAL_METRICS['test_mae']}"
        )

        print(
            f"Test RMSE: "
            f"{FINAL_METRICS['test_rmse']}"
        )

        print(
            f"Test R2: "
            f"{FINAL_METRICS['test_r2']}"
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()