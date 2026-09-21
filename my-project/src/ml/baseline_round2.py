"""Leak-free baseline round 2 using raw House Prices data and CV pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "raw" / "train.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "baseline_round2"

ROUND_NAME = "baseline_round2"
TARGET_COLUMN = "SalePrice"
TARGET_LOG_COLUMN = "SalePriceLog"
SPLIT_TYPE = "final_holdout_before_cv"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_RIDGE_ALPHA = 1.0
CV_N_SPLITS = 5

NUMERIC_FEATURES: list[str] = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
]
CATEGORICAL_NONE_FEATURES: list[str] = ["GarageType"]
CATEGORICAL_MISSING_FEATURES: list[str] = ["Neighborhood"]
ORDINAL_FEATURES: list[str] = ["ExterQual", "KitchenQual", "BsmtQual"]
FEATURE_COLUMNS: list[str] = (
    NUMERIC_FEATURES
    + CATEGORICAL_MISSING_FEATURES
    + CATEGORICAL_NONE_FEATURES
    + ORDINAL_FEATURES
)
ORDINAL_QUALITY_ORDER: list[str] = ["Po", "Fa", "TA", "Gd", "Ex"]
MODEL_NAMES: tuple[str, str] = ("linear_regression", "ridge")

type BaselineModel = LinearRegression | Ridge
type BaselinePipeline = Pipeline
type SplitRawData = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that the raw frame contains the Round 2 inputs and target."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw selected features and log target from a validated frame."""
    x_raw = df[FEATURE_COLUMNS].copy()
    sale_price = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
    y_log = pd.Series(np.log1p(sale_price), index=df.index, name=TARGET_LOG_COLUMN)
    return x_raw, y_log


def load_raw_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw House Prices training data and separate selected features."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Raw training data not found: {data_path}")

    df = pd.read_csv(data_path)
    _validate_required_columns(df)
    return _split_features_target(df)


def split_raw_holdout(
    x_raw: pd.DataFrame,
    y_log: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitRawData:
    """Split raw features before fitting any preprocessing."""
    return cast(
        SplitRawData,
        train_test_split(
            x_raw,
            y_log,
            test_size=test_size,
            random_state=random_state,
        ),
    )


def make_cv() -> KFold:
    """Create the shared Round 2 cross-validation splitter."""
    return KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=DEFAULT_RANDOM_STATE)


def _one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder that tolerates unseen categories."""
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor() -> ColumnTransformer:
    """Build the Round 2 preprocessing ColumnTransformer."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_none_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="None"),
            ),
            ("onehot", _one_hot_encoder()),
        ]
    )
    categorical_missing_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    ordinal_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="None"),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        ORDINAL_QUALITY_ORDER.copy()
                        for _ in range(len(ORDINAL_FEATURES))
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat_none", categorical_none_transformer, CATEGORICAL_NONE_FEATURES),
            (
                "cat_missing",
                categorical_missing_transformer,
                CATEGORICAL_MISSING_FEATURES,
            ),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
        ],
        remainder="drop",
    )


def _build_model(model_name: str, ridge_alpha: float) -> BaselineModel:
    """Create one of the Round 2 baseline regression models."""
    if model_name in {"linear", "linear_regression"}:
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=ridge_alpha)
    raise ValueError("model_name must be one of: linear, linear_regression, ridge")


def build_baseline_pipeline(
    model_name: str,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> BaselinePipeline:
    """Combine Round 2 ColumnTransformer preprocessing with a baseline model."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", _build_model(model_name=model_name, ridge_alpha=ridge_alpha)),
        ]
    )


def _cross_validate_rmse_log(
    pipeline: BaselinePipeline,
    x_train_eval: pd.DataFrame,
    y_train_eval: pd.Series,
    cv: KFold,
) -> list[float]:
    """Run CV manually so fold metrics and the shared splitter stay explicit."""
    rmse_scores: list[float] = []

    for train_indices, validation_indices in cv.split(x_train_eval):
        x_fold_train = x_train_eval.iloc[train_indices]
        x_fold_validation = x_train_eval.iloc[validation_indices]
        y_fold_train = y_train_eval.iloc[train_indices]
        y_fold_validation = y_train_eval.iloc[validation_indices]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(x_fold_train, y_fold_train)
        y_pred_log = fold_pipeline.predict(x_fold_validation)
        rmse_scores.append(
            float(
                np.sqrt(
                    mean_squared_error(
                        y_fold_validation,
                        y_pred_log,
                    )
                )
            )
        )

    return rmse_scores


def _metrics_row(
    model_name: str,
    rmse_scores: Iterable[float],
    x_train_eval: pd.DataFrame,
    test_size: float,
    random_state: int,
    ridge_alpha: float,
    timestamp: str,
) -> dict[str, float | int | str | bool]:
    """Attach Round 2 metadata to mean/std CV metrics."""
    scores = np.asarray(list(rmse_scores), dtype=np.float64)
    return {
        "round": ROUND_NAME,
        "model_name": model_name,
        "n_features": len(FEATURE_COLUMNS),
        "target": TARGET_LOG_COLUMN,
        "split_type": SPLIT_TYPE,
        "test_size": test_size,
        "random_state": random_state,
        "train_eval_rows": len(x_train_eval),
        "cv_n_splits": CV_N_SPLITS,
        "cv_shuffle": True,
        "cv_random_state": DEFAULT_RANDOM_STATE,
        "ridge_alpha": ridge_alpha,
        "timestamp": timestamp,
        "rmse_log_mean": float(np.mean(scores)),
        "rmse_log_std": float(np.std(scores)),
    }


def _fold_metric_rows(
    model_name: str,
    rmse_scores: Iterable[float],
    timestamp: str,
) -> list[dict[str, float | int | str]]:
    """Create one traceable metric row per CV fold."""
    return [
        {
            "round": ROUND_NAME,
            "model_name": model_name,
            "fold": fold_number,
            "target": TARGET_LOG_COLUMN,
            "rmse_log": rmse_log,
            "timestamp": timestamp,
        }
        for fold_number, rmse_log in enumerate(rmse_scores, start=1)
    ]


def _write_final_holdout_indices(
    raw_df: pd.DataFrame,
    x_holdout: pd.DataFrame,
    artifacts_dir: Path,
) -> None:
    """Persist final holdout row identities without evaluating predictions."""
    holdout = pd.DataFrame({"index": x_holdout.index})
    if "Id" in raw_df.columns:
        holdout["Id"] = raw_df.loc[x_holdout.index, "Id"].to_numpy()
    holdout.to_csv(artifacts_dir / "final_holdout_indices.csv", index=False)


def run_baseline_round2(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Run leak-free Round 2 baselines and write CV-only metrics artifacts."""
    raw_df = pd.read_csv(Path(data_path))
    _validate_required_columns(raw_df)
    x_raw, y_log = _split_features_target(raw_df)
    x_train_eval, x_holdout, y_train_eval, _ = split_raw_holdout(
        x_raw=x_raw,
        y_log=y_log,
        test_size=test_size,
        random_state=random_state,
    )

    timestamp = datetime.now().isoformat(timespec="seconds")
    cv = make_cv()
    metric_rows: list[dict[str, float | int | str | bool]] = []
    fold_rows: list[dict[str, float | int | str]] = []

    for model_name in MODEL_NAMES:
        pipeline = build_baseline_pipeline(
            model_name=model_name,
            ridge_alpha=ridge_alpha,
        )
        rmse_scores = _cross_validate_rmse_log(
            pipeline=pipeline,
            x_train_eval=x_train_eval,
            y_train_eval=y_train_eval,
            cv=cv,
        )
        metric_rows.append(
            _metrics_row(
                model_name=model_name,
                rmse_scores=rmse_scores,
                x_train_eval=x_train_eval,
                test_size=test_size,
                random_state=random_state,
                ridge_alpha=ridge_alpha,
                timestamp=timestamp,
            )
        )
        fold_rows.extend(
            _fold_metric_rows(
                model_name=model_name,
                rmse_scores=rmse_scores,
                timestamp=timestamp,
            )
        )

    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    _write_final_holdout_indices(
        raw_df=raw_df,
        x_holdout=x_holdout,
        artifacts_dir=output_dir,
    )

    return metrics_df


if __name__ == "__main__":
    print(run_baseline_round2().to_string(index=False))
