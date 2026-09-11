"""Baseline round 1 training on preprocessed House Prices features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "artifacts" / "preprocessing" / "processed_train.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "baseline_round1"

ROUND_NAME = "baseline_round1"
TARGET_COLUMN = "SalePrice"
TARGET_LOG_COLUMN = "SalePriceLog"
SPLIT_TYPE = "single_holdout"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_RIDGE_ALPHA = 1.0

type BaselineModel = LinearRegression | Ridge
type SplitData = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def load_processed_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load preprocessed training data and separate features from log target."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed training data not found: {data_path}")

    df = pd.read_csv(data_path)
    if TARGET_LOG_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_LOG_COLUMN}")

    excluded_columns = [
        column for column in [TARGET_COLUMN, TARGET_LOG_COLUMN] if column in df.columns
    ]
    x = df.drop(columns=excluded_columns).copy()
    y = pd.to_numeric(df[TARGET_LOG_COLUMN], errors="raise").rename(TARGET_LOG_COLUMN)

    return x, y


def split_processed_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitData:
    """Split processed features and log target into train and holdout sets."""
    return cast(
        SplitData,
        train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
        ),
    )


def train_baseline_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> dict[str, BaselineModel]:
    """Fit LinearRegression and Ridge baseline models."""
    models: dict[str, BaselineModel] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=ridge_alpha),
    }

    for model in models.values():
        model.fit(x_train, y_train)

    return models


def evaluate_regression_model(
    model: BaselineModel,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted baseline on log and original price scales."""
    y_true_log = np.asarray(y_test, dtype=np.float64)
    y_pred_log = np.asarray(model.predict(x_test), dtype=np.float64)
    y_true_price = np.expm1(y_true_log)
    y_pred_price = np.expm1(y_pred_log)

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_price": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "mae_price": float(mean_absolute_error(y_true_price, y_pred_price)),
        "r2_price": float(r2_score(y_true_price, y_pred_price)),
    }


def _metrics_row(
    model_name: str,
    metrics: dict[str, float],
    n_features: int,
    test_size: float,
    random_state: int,
    ridge_alpha: float,
    timestamp: str,
) -> dict[str, float | int | str]:
    """Attach round metadata to one model metrics dictionary."""
    return {
        "round": ROUND_NAME,
        "model_name": model_name,
        "n_features": n_features,
        "target": TARGET_LOG_COLUMN,
        "split_type": SPLIT_TYPE,
        "test_size": test_size,
        "random_state": random_state,
        "ridge_alpha": ridge_alpha,
        "timestamp": timestamp,
        **metrics,
    }


def _save_model_artifacts(
    models: dict[str, BaselineModel],
    artifacts_dir: Path,
) -> None:
    """Persist fitted baseline models using stable round-1 filenames."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        joblib.dump(model, artifacts_dir / f"{model_name}.pkl")


def run_baseline_round1(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Run baseline round 1 training, evaluation, and artifact writing."""
    x, y = load_processed_training_data(Path(data_path))
    x_train, x_test, y_train, y_test = split_processed_data(
        x=x,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )
    models = train_baseline_models(
        x_train=x_train,
        y_train=y_train,
        ridge_alpha=ridge_alpha,
    )

    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = [
        _metrics_row(
            model_name=model_name,
            metrics=evaluate_regression_model(model, x_test, y_test),
            n_features=x.shape[1],
            test_size=test_size,
            random_state=random_state,
            ridge_alpha=ridge_alpha,
            timestamp=timestamp,
        )
        for model_name, model in models.items()
    ]
    metrics_df = pd.DataFrame(rows)

    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    _save_model_artifacts(models=models, artifacts_dir=output_dir)

    return metrics_df
