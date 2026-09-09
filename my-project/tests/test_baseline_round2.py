from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml import baseline_round2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _pipeline_is_fitted(pipeline: Pipeline) -> bool:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    return hasattr(preprocessor, "transformers_") or hasattr(model, "coef_")


def _full_round2_feature_frame(neighborhood: list[str | float]) -> pd.DataFrame:
    n_rows = len(neighborhood)
    data: dict[str, object] = {
        col: [1.0, 2.0, 3.0, 4.0][:n_rows] for col in baseline_round2.NUMERIC_FEATURES
    }
    data.update(
        {
            "GarageType": ["Attchd", "Detchd", "BuiltIn", "Attchd"][:n_rows],
            "Neighborhood": neighborhood,
            "ExterQual": ["TA", "Gd", "Ex", "Fa"][:n_rows],
            "KitchenQual": ["TA", "Gd", "Ex", "Fa"][:n_rows],
            "BsmtQual": ["TA", "Gd", "Ex", "Fa"][:n_rows],
        }
    )
    return pd.DataFrame(data, columns=baseline_round2.FEATURE_COLUMNS)


@pytest.fixture
def raw_training_data() -> pd.DataFrame:
    """Create compact raw data covering the selected Round 2 feature groups."""
    sale_prices = np.array(
        [
            150000,
            175000,
            210000,
            240000,
            265000,
            305000,
            325000,
            360000,
            185000,
            290000,
            410000,
            135000,
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "Id": range(1, 13),
            "OverallQual": [5, 6, 7, 7, 8, 8, 9, 9, 6, 8, 10, 4],
            "GrLivArea": [
                1000,
                1200,
                1450,
                1500,
                1600,
                1800,
                2000,
                2200,
                1300,
                1750,
                2400,
                900,
            ],
            "GarageCars": [1, 1, 2, 2, 2, 2, 3, 3, np.nan, 2, 3, 1],
            "GarageArea": [
                240,
                260,
                400,
                430,
                500,
                520,
                700,
                740,
                np.nan,
                510,
                820,
                200,
            ],
            "TotalBsmtSF": [
                700,
                760,
                820,
                900,
                980,
                1050,
                1200,
                1350,
                np.nan,
                990,
                1500,
                0,
            ],
            "1stFlrSF": [
                700,
                760,
                820,
                900,
                980,
                1050,
                1200,
                1350,
                840,
                990,
                1500,
                650,
            ],
            "FullBath": [1, 1, 2, 2, 2, 2, 3, 3, 1, 2, 3, 1],
            "TotRmsAbvGrd": [5, 6, 6, 7, 7, 8, 8, 9, 6, 8, 10, 4],
            "YearBuilt": [
                1950,
                1960,
                1975,
                1985,
                1995,
                2000,
                2005,
                2010,
                1970,
                2002,
                2015,
                1940,
            ],
            "YearRemodAdd": [
                1970,
                1980,
                1990,
                1995,
                2001,
                2004,
                2008,
                2012,
                1985,
                2006,
                2018,
                1960,
            ],
            "Neighborhood": [
                "NAmes",
                "NAmes",
                "CollgCr",
                np.nan,
                "CollgCr",
                "NridgHt",
                "NridgHt",
                "StoneBr",
                "NAmes",
                "CollgCr",
                "StoneBr",
                "OldTown",
            ],
            "GarageType": [
                "Detchd",
                "Attchd",
                "Attchd",
                np.nan,
                "BuiltIn",
                "Attchd",
                "BuiltIn",
                "Attchd",
                np.nan,
                "Attchd",
                "BuiltIn",
                "Detchd",
            ],
            "ExterQual": [
                "TA",
                "TA",
                "Gd",
                "Gd",
                "Gd",
                "Ex",
                "Ex",
                "Ex",
                "TA",
                "Gd",
                "Ex",
                "Fa",
            ],
            "KitchenQual": [
                "TA",
                "TA",
                "Gd",
                "Gd",
                "Gd",
                "Ex",
                "Ex",
                "Ex",
                np.nan,
                "Gd",
                "Ex",
                "Fa",
            ],
            "BsmtQual": [
                "TA",
                "TA",
                "Gd",
                np.nan,
                "Gd",
                "Ex",
                "Ex",
                "Ex",
                "TA",
                "Gd",
                "Ex",
                np.nan,
            ],
            "SalePrice": sale_prices,
        }
    )


def test_load_raw_training_data_selects_raw_features_and_log_target(
    raw_training_data: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "train.csv"
    raw_training_data.to_csv(data_path, index=False)

    x_raw, y_log = baseline_round2.load_raw_training_data(data_path)

    assert list(x_raw.columns) == baseline_round2.FEATURE_COLUMNS
    assert "SalePrice" not in x_raw.columns
    assert y_log.name == baseline_round2.TARGET_LOG_COLUMN
    np.testing.assert_allclose(
        y_log.to_numpy(), np.log1p(raw_training_data["SalePrice"])
    )


def test_load_raw_training_data_requires_selected_columns(
    raw_training_data: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "train.csv"
    raw_training_data.drop(columns=["KitchenQual"]).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="KitchenQual"):
        baseline_round2.load_raw_training_data(data_path)


def test_split_raw_holdout_preserves_final_holdout_outside_train_eval(
    raw_training_data: pd.DataFrame,
) -> None:
    x_raw = raw_training_data[baseline_round2.FEATURE_COLUMNS].copy()
    y_log = np.log1p(raw_training_data["SalePrice"])

    x_train_eval, x_holdout, y_train_eval, y_holdout = (
        baseline_round2.split_raw_holdout(
            x_raw,
            y_log,
            test_size=0.25,
            random_state=7,
        )
    )

    assert len(x_train_eval) == 9
    assert len(x_holdout) == 3
    assert set(x_train_eval.index).isdisjoint(set(x_holdout.index))
    assert set(y_train_eval.index).isdisjoint(set(y_holdout.index))


def test_make_cv_uses_shared_round2_configuration() -> None:
    cv = baseline_round2.make_cv()
    cv_runtime = cast(Any, cv)

    assert isinstance(cv, KFold)
    assert cv_runtime.n_splits == 5
    assert cv_runtime.shuffle is True
    assert cv_runtime.random_state == 42


def test_build_baseline_pipeline_uses_column_transformer_only_imputation() -> None:
    pipeline = baseline_round2.build_baseline_pipeline("ridge", ridge_alpha=2.5)
    model = cast(Any, pipeline.named_steps["model"])

    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    assert isinstance(pipeline.named_steps["model"], Ridge)
    assert model.alpha == 2.5

    preprocessor = pipeline.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer)
    preprocessor_runtime = cast(Any, preprocessor)
    transformer_names = [name for name, _, _ in preprocessor_runtime.transformers]
    assert transformer_names == ["num", "cat_none", "cat_missing", "ord"]
    transformers = {
        name: transformer for name, transformer, _ in preprocessor_runtime.transformers
    }

    num_pipeline = transformers["num"]
    assert isinstance(num_pipeline, Pipeline)
    assert isinstance(num_pipeline.named_steps["imputer"], SimpleImputer)
    assert cast(Any, num_pipeline.named_steps["imputer"]).strategy == "median"
    assert isinstance(num_pipeline.named_steps["scaler"], StandardScaler)

    cat_none_pipeline = transformers["cat_none"]
    cat_none_imputer = cast(Any, cat_none_pipeline.named_steps["imputer"])
    assert cat_none_imputer.strategy == "constant"
    assert cat_none_imputer.fill_value == "None"
    assert isinstance(cat_none_pipeline.named_steps["onehot"], OneHotEncoder)

    cat_missing_pipeline = transformers["cat_missing"]
    cat_missing_imputer = cast(Any, cat_missing_pipeline.named_steps["imputer"])
    assert cat_missing_imputer.strategy == "most_frequent"
    assert isinstance(cat_missing_pipeline.named_steps["onehot"], OneHotEncoder)

    ord_pipeline = transformers["ord"]
    ordinal = ord_pipeline.named_steps["ordinal"]
    assert isinstance(ordinal, OrdinalEncoder)
    assert cast(Any, ordinal).categories == [
        baseline_round2.ORDINAL_QUALITY_ORDER,
        baseline_round2.ORDINAL_QUALITY_ORDER,
        baseline_round2.ORDINAL_QUALITY_ORDER,
    ]


def test_build_baseline_pipeline_supports_linear_regression() -> None:
    pipeline = baseline_round2.build_baseline_pipeline("linear")

    assert isinstance(pipeline.named_steps["model"], LinearRegression)


def test_cross_validate_rmse_log_does_not_fit_original_pipeline(
    raw_training_data: pd.DataFrame,
) -> None:
    x_raw = raw_training_data[baseline_round2.FEATURE_COLUMNS].copy()
    y_log = np.log1p(raw_training_data["SalePrice"])
    x_train_eval, _, y_train_eval, _ = baseline_round2.split_raw_holdout(
        x_raw,
        y_log,
        test_size=0.25,
        random_state=7,
    )
    pipeline = baseline_round2.build_baseline_pipeline("ridge")

    fitted_before = _pipeline_is_fitted(pipeline)
    scores = baseline_round2._cross_validate_rmse_log(
        pipeline=pipeline,
        x_train_eval=x_train_eval,
        y_train_eval=y_train_eval,
        cv=baseline_round2.make_cv(),
    )
    fitted_after = _pipeline_is_fitted(pipeline)

    assert fitted_before is False
    assert len(scores) == 5
    assert np.isfinite(scores).all()
    assert fitted_after is False


def test_bsmtqual_none_is_encoded_as_unknown_negative_one() -> None:
    pipeline = baseline_round2.build_baseline_pipeline("linear")
    preprocessor = pipeline.named_steps["preprocessor"]

    df = pd.DataFrame(
        {
            **{col: [1.0, 2.0, 3.0] for col in baseline_round2.NUMERIC_FEATURES},
            "GarageType": ["Attchd", np.nan, "Detchd"],
            "Neighborhood": ["NAmes", "CollgCr", "NAmes"],
            "ExterQual": ["Po", "TA", "Ex"],
            "KitchenQual": ["Fa", "Gd", "Ex"],
            "BsmtQual": [np.nan, "TA", "Ex"],
        }
    )

    transformed = preprocessor.fit_transform(df)
    ordinal_slice = preprocessor.output_indices_["ord"]
    ordinal_values = transformed[:, ordinal_slice]

    np.testing.assert_array_equal(ordinal_values[0], np.array([0.0, 1.0, -1.0]))


def test_neighborhood_nan_is_imputed_with_train_most_frequent_category() -> None:
    preprocessor = baseline_round2.build_preprocessor()
    df = _full_round2_feature_frame(["NAmes", np.nan, "CollgCr", "NAmes"])

    transformed = preprocessor.fit_transform(df)
    cat_missing_pipeline = preprocessor.named_transformers_["cat_missing"]
    imputer = cat_missing_pipeline.named_steps["imputer"]
    onehot = cat_missing_pipeline.named_steps["onehot"]
    cat_missing_slice = preprocessor.output_indices_["cat_missing"]
    nan_row_onehot = transformed[1, cat_missing_slice]
    categories = list(onehot.categories_[0])

    assert imputer.statistics_[0] == "NAmes"
    assert categories == ["CollgCr", "NAmes"]
    assert nan_row_onehot[categories.index("NAmes")] == 1.0
    assert nan_row_onehot[categories.index("CollgCr")] == 0.0


def test_run_baseline_round2_writes_cv_metrics_without_holdout_scores(
    raw_training_data: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "train.csv"
    artifacts_dir = tmp_path / "baseline_round2"
    raw_training_data.to_csv(data_path, index=False)

    metrics = baseline_round2.run_baseline_round2(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        test_size=0.25,
        random_state=7,
        ridge_alpha=2.5,
    )

    saved_metrics = pd.read_csv(artifacts_dir / "metrics.csv")
    fold_metrics = pd.read_csv(artifacts_dir / "fold_metrics.csv")
    holdout_indices = pd.read_csv(artifacts_dir / "final_holdout_indices.csv")

    assert set(metrics["model_name"]) == {"linear_regression", "ridge"}
    assert set(saved_metrics["round"]) == {"baseline_round2"}
    assert set(saved_metrics["split_type"]) == {"final_holdout_before_cv"}
    assert set(saved_metrics["cv_n_splits"]) == {5}
    assert set(saved_metrics["cv_shuffle"]) == {True}
    assert set(saved_metrics["cv_random_state"]) == {42}
    assert {"rmse_log_mean", "rmse_log_std"}.issubset(saved_metrics.columns)
    assert not any(column == "rmse_log" for column in saved_metrics.columns)
    assert set(fold_metrics["fold"]) == {1, 2, 3, 4, 5}
    assert set(fold_metrics["model_name"]) == {"linear_regression", "ridge"}
    assert "rmse_log" in fold_metrics.columns
    assert list(holdout_indices.columns) == ["index", "Id"]
    assert len(holdout_indices) == 3
    assert np.isfinite(saved_metrics["rmse_log_mean"]).all()
    assert np.isfinite(saved_metrics["rmse_log_std"]).all()
