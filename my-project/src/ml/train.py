"""Production training script for the Kaggle House Prices model."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    HousePricesMissingValueImputer,
    build_preprocessing_pipeline,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "raw" / "train.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DEFAULT_MODEL_PATH = PROJECT_DIR / "model.pkl"

MODEL_CHOICES = ("linear", "ridge", "lasso")
DEFAULT_MODEL = "ridge"
DEFAULT_ALPHA = 1.0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
LASSO_MAX_ITER = 5000


def configure_logging(level: str = "INFO") -> None:
    """Configure basic logging for command-line runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data, select session features, and log-transform the target."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    x_raw = df[FEATURE_COLUMNS].copy()
    y_log = pd.Series(np.log1p(df[TARGET_COLUMN]), index=df.index, name=TARGET_COLUMN)

    LOGGER.info("Loaded data from %s with shape %s", data_path, df.shape)
    return x_raw, y_log


def build_model(
    model_name: str, alpha: float, warn_alpha: bool = False
) -> LinearRegression | Ridge | Lasso:
    """Create a regression model from CLI options."""
    if model_name == "linear":
        if warn_alpha:
            LOGGER.warning("alpha is ignored for linear regression.")
        return LinearRegression()

    if model_name == "ridge":
        return Ridge(alpha=alpha)

    if model_name == "lasso":
        return Lasso(alpha=alpha, max_iter=LASSO_MAX_ITER)

    valid_values = ", ".join(MODEL_CHOICES)
    raise ValueError(f"Unknown model '{model_name}'. Expected one of: {valid_values}.")

#Pipeline for preprocessing and regression
def build_pipeline(
    model_name: str = DEFAULT_MODEL,
    alpha: float = DEFAULT_ALPHA,
    warn_alpha: bool = False,
) -> Pipeline:
    """Build a train-only preprocessing and regression pipeline."""
    preprocessor = build_preprocessing_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=NOMINAL_FEATURES,
        ordinal_features=ORDINAL_FEATURES,
    )
    model = build_model(model_name=model_name, alpha=alpha, warn_alpha=warn_alpha)

    return Pipeline(
        steps=[
            ("missing_values", HousePricesMissingValueImputer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

#Train model on training data
def train(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the full sklearn pipeline on train data."""
    pipeline.fit(x_train, y_train)
    LOGGER.info("Training completed on %s rows", len(x_train))
    return pipeline

# Evaluate the trained model on the holdout set
def evaluate(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate the pipeline on the holdout split using log-target metrics."""
    y_pred_log = pipeline.predict(x_test)
    metrics = {
        "rmse_log": float(np.sqrt(mean_squared_error(y_test, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_test, y_pred_log)),
        "r2": float(r2_score(y_test, y_pred_log)),
    }
    LOGGER.info(
        "Evaluation completed: rmse_log=%.4f mae_log=%.4f r2=%.4f",
        metrics["rmse_log"],
        metrics["mae_log"],
        metrics["r2"],
    )
    return metrics


def _format_alpha(alpha: float) -> str:
    """Return a readable alpha value for filenames and experiment logs."""
    return str(alpha)


def _unique_path(path: Path) -> Path:
    """Return a non-existing path by adding a numeric suffix if needed."""
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _atomic_dump(pipeline: Pipeline, model_path: Path) -> None:
    """Replace a model artifact only after serialization succeeds."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=model_path.parent,
        prefix=f".{model_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        joblib.dump(pipeline, temporary_path)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict[str, float],
    artifacts_dir: Path,
    experiments_path: Path,
    model_name: str,
    alpha: float,
    data_path: Path,
    test_size: float,
    random_state: int,
    train_rows: int,
    test_rows: int,
    model_path: Path | None = None,
    timestamp: datetime | None = None,
) -> tuple[Path, Path]:
    """Save the fitted model and append one experiment row."""
    run_timestamp = timestamp or datetime.now()
    artifacts_dir = Path(artifacts_dir)
    experiments_path = Path(experiments_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        timestamp_label = run_timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"model_{model_name}_a{_format_alpha(alpha)}_{timestamp_label}.pkl"
        model_path = _unique_path(artifacts_dir / filename)
    else:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    _atomic_dump(pipeline, model_path)

    row = {
        "timestamp": run_timestamp.isoformat(timespec="seconds"),
        "model": model_name,
        "alpha": alpha,
        "data_path": str(Path(data_path).resolve()),
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "rmse_log": metrics["rmse_log"],
        "mae_log": metrics["mae_log"],
        "r2": metrics["r2"],
        "model_artifact_path": str(model_path.resolve()),
    }
    should_write_header = not experiments_path.exists()
    pd.DataFrame([row]).to_csv(
        experiments_path,
        mode="a",
        header=should_write_header,
        index=False,
    )

    LOGGER.info("Saved model artifact to %s", model_path)
    LOGGER.info("Appended experiment metrics to %s", experiments_path)
    return model_path, experiments_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    argv_list = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--experiments-path", type=Path, default=None)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv_list)
    args.alpha_provided = any(
        arg == "--alpha" or arg.startswith("--alpha=") for arg in argv_list
    )
    return args

#Split the data into training and holdout set
def main(argv: Sequence[str] | None = None) -> dict[str, float | str]:
    """Run the full training experiment from the command line."""
    args = parse_args(argv)
    configure_logging(args.log_level)

    x_raw, y_log = load_data(Path(args.data_path))
    x_train, x_test, y_train, y_test = train_test_split(
        x_raw,
        y_log,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    LOGGER.info(
        "Split data into %s train rows and %s test rows",
        len(x_train),
        len(x_test),
    )

    pipeline = build_pipeline(
        model_name=args.model,
        alpha=args.alpha,
        warn_alpha=args.alpha_provided,
    )
    fitted_pipeline = train(pipeline=pipeline, x_train=x_train, y_train=y_train)
    metrics = evaluate(pipeline=fitted_pipeline, x_test=x_test, y_test=y_test)

    artifacts_dir = Path(args.artifacts_dir)
    experiments_path = (
        Path(args.experiments_path)
        if args.experiments_path is not None
        else artifacts_dir / "experiments.csv"
    )
#Save the train model and append one experiment row
    model_path, saved_experiments_path = save_artifacts(
        pipeline=fitted_pipeline,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
        experiments_path=experiments_path,
        model_name=args.model,
        alpha=args.alpha,
        data_path=Path(args.data_path),
        test_size=args.test_size,
        random_state=args.random_state,
        train_rows=len(x_train),
        test_rows=len(x_test),
        model_path=args.model_path,
    )

    return {
        **metrics,
        "model_artifact_path": str(model_path),
        "experiments_path": str(saved_experiments_path),
    }


if __name__ == "__main__":
    main()
