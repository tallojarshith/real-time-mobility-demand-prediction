import os
import joblib

from src.utils.logger import get_logger


logger = get_logger(__name__)


MODEL_PATH = "artifacts/final_extra_trees_model.joblib"
FEATURES_PATH = "artifacts/feature_columns.joblib"


def load_model_and_features(
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH
):
    """
    Load the trained demand forecasting model and its
    expected feature schema.

    Returns
    -------
    model
        Trained scikit-learn model.

    feature_columns : list
        Ordered list of features expected by the model.
    """

    logger.info("Loading trained demand forecasting model")

    # -----------------------------------------------------
    # Validate artifact paths
    # -----------------------------------------------------

    if not os.path.exists(model_path):

        logger.error(
            "Model artifact not found: %s",
            model_path
        )

        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    if not os.path.exists(features_path):

        logger.error(
            "Feature schema artifact not found: %s",
            features_path
        )

        raise FileNotFoundError(
            f"Feature schema artifact not found: {features_path}"
        )

    # -----------------------------------------------------
    # Load artifacts
    # -----------------------------------------------------

    model = joblib.load(model_path)

    feature_columns = joblib.load(
        features_path
    )

    feature_columns = list(feature_columns)

    # -----------------------------------------------------
    # Validate feature count
    # -----------------------------------------------------

    if hasattr(model, "n_features_in_"):

        if model.n_features_in_ != len(feature_columns):

            logger.error(
                "Feature count mismatch: "
                "model expects %d but schema contains %d",
                model.n_features_in_,
                len(feature_columns)
            )

            raise ValueError(
                "Model and feature schema have "
                "different feature counts"
            )

    # -----------------------------------------------------
    # Validate exact feature names and order
    # -----------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

        if model_features != feature_columns:

            logger.error(
                "Feature schema does not match "
                "model feature names/order"
            )

            raise ValueError(
                "Feature schema does not match "
                "the trained model"
            )

    logger.info(
        "Model loaded successfully with %d features",
        len(feature_columns)
    )

    return model, feature_columns