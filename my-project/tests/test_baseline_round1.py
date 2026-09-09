from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml import baseline_round1
from sklearn.linear_model import LinearRegression, Ridge


@pytest.fixture
def processed_training_data() -> pd.DataFrame:
    """Create compact processed training data for baseline round 1 tests."""
    sale_prices = np.array(
        [150000, 175000, 210000, 240000, 265000, 305000, 325000, 360000],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "num__OverallQual": [-1.2, -0.8, -0.2, 0.1, 0.4, 0.8, 1.0, 1.4],
            "num__GrLivArea": [-1.0, -0.6, -0.1, 0.2, 0.5, 0.9, 1.1, 1.5],
            "cat__Neighborhood_CollgCr": [1, 1, 0, 0, 1, 0, 0, 1],
            "ord__KitchenQual": [2, 2, 3, 3, 3, 4, 4, 4],
            "SalePrice": sale_prices,
            "SalePriceLog": np.log1p(sale_prices),
        }
    )


def test_load_processed_training_data_requires_log_target(
    processed_training_data: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "processed_train.csv"
    processed_training_data.drop(columns=["SalePriceLog"]).to_csv(
        data_path, index=False
    )

    with pytest.raises(ValueError, match="SalePriceLog"):
        baseline_round1.load_processed_training_data(data_path)


def test_load_processed_training_data_excludes_target_columns(
    processed_training_data: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "processed_train.csv"
    processed_training_data.to_csv(data_path, index=False)

    x, y = baseline_round1.load_processed_training_data(data_path)

    assert "SalePrice" not in x.columns
    assert "SalePriceLog" not in x.columns
    assert y.name == "SalePriceLog"
    assert len(x) == len(y)


def test_split_processed_data_accepts_test_size_and_random_state(
    processed_training_data: pd.DataFrame,
) -> None:
    x = processed_training_data.drop(columns=["SalePrice", "SalePriceLog"])
    y = processed_training_data["SalePriceLog"]

    first_split = baseline_round1.split_processed_data(
        x, y, test_size=0.25, random_state=7
    )
    second_split = baseline_round1.split_processed_data(
        x, y, test_size=0.25, random_state=7
    )
    x_train, x_test, y_train, y_test = first_split

    assert len(x_train) == 6
    assert len(x_test) == 2
    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])
    pd.testing.assert_series_equal(y_train, second_split[2])
    pd.testing.assert_series_equal(y_test, second_split[3])


def test_train_baseline_models_trains_linear_regression_and_ridge(
    processed_training_data: pd.DataFrame,
) -> None:
    x = processed_training_data.drop(columns=["SalePrice", "SalePriceLog"])
    y = processed_training_data["SalePriceLog"]
    x_train, _, y_train, _ = baseline_round1.split_processed_data(
        x, y, test_size=0.25, random_state=42
    )

    models = baseline_round1.train_baseline_models(
        x_train, y_train, ridge_alpha=2.5
    )

    assert set(models) == {"linear_regression", "ridge"}
    assert isinstance(models["linear_regression"], LinearRegression)
    assert isinstance(models["ridge"], Ridge)
    assert models["ridge"].alpha == 2.5
    assert hasattr(models["linear_regression"], "coef_")
    assert hasattr(models["ridge"], "coef_")


def test_evaluate_regression_model_returns_finite_log_and_price_metrics(
    processed_training_data: pd.DataFrame,
) -> None:
    x = processed_training_data.drop(columns=["SalePrice", "SalePriceLog"])
    y = processed_training_data["SalePriceLog"]
    x_train, x_test, y_train, y_test = baseline_round1.split_processed_data(
        x, y, test_size=0.25, random_state=42
    )
    model = baseline_round1.train_baseline_models(x_train, y_train)["ridge"]

    metrics = baseline_round1.evaluate_regression_model(model, x_test, y_test)

    assert set(metrics) == {
        "rmse_log",
        "mae_log",
        "r2_log",
        "rmse_price",
        "mae_price",
        "r2_price",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_run_baseline_round1_writes_metrics_and_model_artifacts(
    processed_training_data: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "processed_train.csv"
    artifacts_dir = tmp_path / "baseline_round1"
    processed_training_data.to_csv(data_path, index=False)

    metrics = baseline_round1.run_baseline_round1(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        test_size=0.25,
        random_state=7,
        ridge_alpha=2.5,
    )

    metrics_path = artifacts_dir / "metrics.csv"
    saved_metrics = pd.read_csv(metrics_path)

    assert metrics_path.exists()
    assert (artifacts_dir / "linear_regression.pkl").exists()
    assert (artifacts_dir / "ridge.pkl").exists()
    assert set(saved_metrics["model_name"]) == {"linear_regression", "ridge"}
    assert set(metrics["model_name"]) == {"linear_regression", "ridge"}
    for column in [
        "round",
        "n_features",
        "timestamp",
        "split_type",
        "test_size",
        "random_state",
        "ridge_alpha",
    ]:
        assert column in saved_metrics.columns
    assert set(saved_metrics["round"]) == {"baseline_round1"}
    assert set(saved_metrics["n_features"]) == {4}
    assert set(saved_metrics["split_type"]) == {"single_holdout"}
