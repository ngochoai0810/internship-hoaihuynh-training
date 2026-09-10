import hashlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml import train as train_module
from ml.preprocessing import (
    HousePricesMissingValueImputer,
    build_preprocessing_pipeline,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@pytest.fixture
def reduced_training_data() -> pd.DataFrame:
    """Create selected 15-field raw training data for train.py tests."""
    return pd.DataFrame(
        {
            "OverallQual": [7, 6, 8, 5, 8, 5, 9, 7],
            "GrLivArea": [1710, 1262, 1786, 1100, 1980, 1150, 2100, 1500],
            "GarageCars": [2.0, 2.0, 3.0, 1.0, 2.0, 1.0, 3.0, 2.0],
            "GarageArea": [548.0, 460.0, 836.0, 300.0, 600.0, 280.0, 900.0, 500.0],
            "TotalBsmtSF": [856, 1262, 920, 756, 1145, 796, 1686, 1107],
            "1stFlrSF": [856, 1262, 920, 756, 1145, 796, 1686, 1107],
            "FullBath": [2, 2, 2, 1, 2, 1, 3, 2],
            "TotRmsAbvGrd": [8, 6, 6, 5, 8, 5, 10, 7],
            "YearBuilt": [2003, 1976, 2001, 1965, 2005, 1960, 2010, 1998],
            "YearRemodAdd": [2003, 1976, 2002, 1990, 2005, 1980, 2011, 1999],
            "Neighborhood": [
                "CollgCr",
                "Veenker",
                "Crawfor",
                "NAmes",
                "NoRidge",
                "OldTown",
                "NridgHt",
                "Somerst",
            ],
            "GarageType": [
                "Attchd",
                np.nan,
                "Detchd",
                "BuiltIn",
                "Attchd",
                "Detchd",
                "Attchd",
                "BuiltIn",
            ],
            "ExterQual": ["Gd", "TA", "TA", "Ex", "Gd", "TA", "Fa", "Gd"],
            "KitchenQual": ["Gd", "TA", "Gd", "TA", "Ex", "TA", "Ex", "Gd"],
            "BsmtQual": ["Gd", "TA", "Gd", "TA", "Ex", "TA", np.nan, "Gd"],
            "SalePrice": [
                208500,
                181500,
                223500,
                140000,
                250000,
                143000,
                307000,
                200000,
            ],
        }
    )


def _contains_simple_imputer(estimator: object) -> bool:
    if isinstance(estimator, SimpleImputer):
        return True

    if isinstance(estimator, Pipeline):
        return any(_contains_simple_imputer(step) for _, step in estimator.steps)

    if isinstance(estimator, ColumnTransformer):
        return any(
            not isinstance(transformer, str) and _contains_simple_imputer(transformer)
            for _, transformer, _ in estimator.transformers
        )

    return False


def test_missing_value_imputer_uses_train_median_for_test() -> None:
    train_df = pd.DataFrame(
        {
            "GrLivArea": [1000.0, 1500.0, 2000.0],
            "GarageArea": [200.0, 400.0, 600.0],
            "Neighborhood": ["NAmes", "CollgCr", "Somerst"],
            "GarageType": ["Attchd", "Detchd", "BuiltIn"],
            "ExterQual": ["TA", "Gd", "Ex"],
            "BsmtQual": ["TA", "Gd", np.nan],
        }
    )
    test_df = pd.DataFrame(
        {
            "GrLivArea": [np.nan, 2500.0, 3000.0],
            "GarageArea": [100.0, 200.0, 300.0],
            "Neighborhood": [np.nan, "NAmes", "Somerst"],
            "GarageType": [np.nan, "Attchd", "Detchd"],
            "ExterQual": ["TA", "Gd", "Ex"],
            "BsmtQual": [np.nan, "TA", "Gd"],
        }
    )

    imputer = HousePricesMissingValueImputer().fit(train_df)
    transformed = imputer.transform(test_df)

    assert transformed["GrLivArea"].iloc[0] == 1500.0
    assert transformed["GarageType"].iloc[0] == "None"
    assert transformed["BsmtQual"].iloc[0] == "None"


def test_build_pipeline_fits_without_engineered_columns(
    reduced_training_data: pd.DataFrame,
) -> None:
    x_raw = reduced_training_data[train_module.FEATURE_COLUMNS]
    y_log = np.log1p(reduced_training_data["SalePrice"])

    pipeline = train_module.build_pipeline(model_name="ridge", alpha=1.0)
    pipeline.fit(x_raw, y_log)
    predictions = pipeline.predict(x_raw.iloc[:3])

    assert predictions.shape == (3,)


def test_build_pipeline_has_single_imputation_layer() -> None:
    pipeline = train_module.build_pipeline()

    assert isinstance(
        pipeline.named_steps["missing_values"], HousePricesMissingValueImputer
    )
    assert not _contains_simple_imputer(pipeline)


def test_ordinal_encoder_uses_semantic_quality_order() -> None:
    df = pd.DataFrame({"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"]})
    preprocessor = build_preprocessing_pipeline(
        numeric_features=[],
        categorical_features=[],
        ordinal_features=["ExterQual"],
    )

    encoded = preprocessor.fit_transform(df)

    np.testing.assert_array_equal(encoded.ravel(), np.array([0, 1, 2, 3, 4]))


def test_invalid_model_rejected_by_cli() -> None:
    with pytest.raises(SystemExit):
        train_module.parse_args(["--model", "elasticnet"])


def test_linear_model_warns_when_alpha_is_provided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=train_module.LOGGER.name):
        train_module.build_pipeline(model_name="linear", alpha=99.0, warn_alpha=True)

    assert "alpha is ignored for linear regression" in caplog.text


def test_save_artifacts_creates_unique_model_and_appends_experiments(
    reduced_training_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    x_raw = reduced_training_data[train_module.FEATURE_COLUMNS]
    y_log = np.log1p(reduced_training_data["SalePrice"])
    pipeline = train_module.build_pipeline(model_name="ridge", alpha=1.0)
    pipeline.fit(x_raw, y_log)

    artifacts_dir = tmp_path / "artifacts"
    experiments_path = artifacts_dir / "experiments.csv"
    metrics = {"rmse_log": 0.2, "mae_log": 0.1, "r2": 0.8}
    fixed_timestamp = datetime(2026, 8, 14, 9, 0, 0)

    first_model_path, _ = train_module.save_artifacts(
        pipeline=pipeline,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
        experiments_path=experiments_path,
        model_name="ridge",
        alpha=1.0,
        data_path=tmp_path / "train.csv",
        test_size=0.2,
        random_state=42,
        train_rows=6,
        test_rows=2,
        timestamp=fixed_timestamp,
    )
    second_model_path, _ = train_module.save_artifacts(
        pipeline=pipeline,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
        experiments_path=experiments_path,
        model_name="ridge",
        alpha=1.0,
        data_path=tmp_path / "train.csv",
        test_size=0.2,
        random_state=42,
        train_rows=6,
        test_rows=2,
        timestamp=fixed_timestamp,
    )

    experiments = pd.read_csv(experiments_path)

    assert first_model_path.exists()
    assert second_model_path.exists()
    assert first_model_path != second_model_path
    assert len(experiments) == 2
    assert set(experiments["model_artifact_path"]) == {
        str(first_model_path.resolve()),
        str(second_model_path.resolve()),
    }


def test_cli_runs_with_temp_csv(
    reduced_training_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "train.csv"
    artifacts_dir = tmp_path / "artifacts"
    model_path = tmp_path / "model.pkl"
    reduced_training_data.to_csv(data_path, index=False)

    result = train_module.main(
        [
            "--data-path",
            str(data_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--model-path",
            str(model_path),
            "--model",
            "ridge",
            "--alpha",
            "1.0",
            "--test-size",
            "0.25",
            "--random-state",
            "42",
            "--log-level",
            "WARNING",
        ]
    )

    assert result["model_artifact_path"] == str(model_path)
    assert model_path.exists()
    assert Path(result["experiments_path"]).exists()
    assert np.isfinite(result["rmse_log"])


def test_cli_defaults_to_stable_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stable_model_path = tmp_path / "model.pkl"
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", stable_model_path)

    args = train_module.parse_args([])

    assert args.model_path == stable_model_path


def test_failed_model_dump_preserves_previous_stable_artifact(
    reduced_training_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    x_raw = reduced_training_data[train_module.FEATURE_COLUMNS]
    y_log = np.log1p(reduced_training_data["SalePrice"])
    pipeline = train_module.build_pipeline(model_name="ridge", alpha=1.0)
    pipeline.fit(x_raw, y_log)
    stable_model_path = tmp_path / "model.pkl"
    stable_model_path.write_bytes(b"previous-valid-model")

    def fail_after_partial_write(model: object, path: Path) -> None:
        Path(path).write_bytes(b"partial-model")
        raise OSError("disk write failed")

    monkeypatch.setattr(train_module.joblib, "dump", fail_after_partial_write)

    with pytest.raises(OSError, match="disk write failed"):
        train_module.save_artifacts(
            pipeline=pipeline,
            metrics={"rmse_log": 0.2, "mae_log": 0.1, "r2": 0.8},
            artifacts_dir=tmp_path / "artifacts",
            experiments_path=tmp_path / "artifacts" / "experiments.csv",
            model_name="ridge",
            alpha=1.0,
            data_path=tmp_path / "train.csv",
            test_size=0.2,
            random_state=42,
            train_rows=6,
            test_rows=2,
            model_path=stable_model_path,
        )

    assert stable_model_path.read_bytes() == b"previous-valid-model"


def test_retraining_overwrites_same_path_with_new_artifact(
    reduced_training_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.pkl"
    artifacts_dir = tmp_path / "artifacts"
    reduced_training_data.to_csv(data_path, index=False)

    common_args = [
        "--data-path",
        str(data_path),
        "--artifacts-dir",
        str(artifacts_dir),
        "--model-path",
        str(model_path),
        "--model",
        "ridge",
        "--test-size",
        "0.25",
        "--log-level",
        "WARNING",
    ]
    first = train_module.main([*common_args, "--alpha", "1.0"])
    first_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    second = train_module.main([*common_args, "--alpha", "10.0"])
    second_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    assert first["model_artifact_path"] == str(model_path)
    assert second["model_artifact_path"] == str(model_path)
    assert first_hash != second_hash
