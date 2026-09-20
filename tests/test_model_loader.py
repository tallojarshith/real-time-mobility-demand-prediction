import pytest

from src.models.model_loader import (
    load_model_and_features
)


def test_load_model_and_features():

    model, feature_columns = (
        load_model_and_features()
    )

    assert model is not None

    assert len(feature_columns) == 22

    assert model.n_features_in_ == 22

    assert list(
        model.feature_names_in_
    ) == feature_columns


def test_missing_model_file():

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found"
    ):

        load_model_and_features(
            model_path="artifacts/nonexistent_model.joblib"
        )


def test_missing_feature_file():

    with pytest.raises(
        FileNotFoundError,
        match="Feature schema artifact not found"
    ):

        load_model_and_features(
            features_path="artifacts/nonexistent_features.joblib"
        )