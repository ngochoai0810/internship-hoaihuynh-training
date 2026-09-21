"""Package the selected model and score the final holdout exactly once."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import baseline_round2, ensemble_round3  # noqa: E402
from ml.runtime import load_model_artifact, predict_price  # noqa: E402
from ml.train import atomic_dump  # noqa: E402
from schemas.prediction import PredictionInput  # noqa: E402

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = baseline_round2.PROJECT_DIR
DEFAULT_DATA_PATH = baseline_round2.DEFAULT_DATA_PATH
DEFAULT_ROUND3_METRICS = PROJECT_DIR / "artifacts" / "ensemble_round3" / "metrics.csv"
DEFAULT_RIDGE_SEARCH = (
    PROJECT_DIR / "artifacts" / "ridge_tuning" / "ridge_alpha_search.csv"
)
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "final"
DEFAULT_MODEL_PATH = PROJECT_DIR / "model.pkl"

SUPPORTED_MODEL_NAME = "ridge"
DEMO_SAMPLE_COUNT = 3
MAX_PLAUSIBLE_PRICE = 10_000_000.0

SELECTION_NOTE = (
    "Round 3 kept Ridge from Round 2. Gradient Boosting reached a lower CV RMSE "
    "than Ridge on all five folds, but the mean improvement stayed below the "
    "Ridge CV standard deviation, which is the pre-registered switching rule in "
    "ensemble_round3.passes_selection_threshold. Gradient Boosting was also the "
    "best of twelve candidates scored on the same folds used for the comparison, "
    "so its advantage may be mildly optimistic. The final holdout number below "
    "is a report-only figure and must not drive any further model choice."
)

# Reverse of runtime.build_model_frame, used to emit snake-case demo payloads.
PAYLOAD_FIELD_BY_COLUMN: dict[str, str] = {
    "OverallQual": "overall_qual",
    "GrLivArea": "gr_liv_area",
    "GarageCars": "garage_cars",
    "GarageArea": "garage_area",
    "TotalBsmtSF": "total_bsmt_sf",
    "1stFlrSF": "first_flr_sf",
    "FullBath": "full_bath",
    "TotRmsAbvGrd": "tot_rms_abv_grd",
    "YearBuilt": "year_built",
    "YearRemodAdd": "year_remod_add",
    "Neighborhood": "neighborhood",
    "GarageType": "garage_type",
    "ExterQual": "exter_qual",
    "KitchenQual": "kitchen_qual",
    "BsmtQual": "bsmt_qual",
}
INTEGER_PAYLOAD_FIELDS = frozenset(
    {
        "overall_qual",
        "full_bath",
        "tot_rms_abv_grd",
        "year_built",
        "year_remod_add",
    }
)


def load_selected_model_name(metrics_path: Path) -> str:
    """Return the Round 3 winner and refuse to package an unsupported model."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Round 3 metrics not found: {metrics_path}")

    model_name = str(ensemble_round3.choose_selected_model(pd.read_csv(metrics_path)))
    if model_name != SUPPORTED_MODEL_NAME:
        raise NotImplementedError(
            f"Round 3 selected '{model_name}', but packaging only supports "
            f"'{SUPPORTED_MODEL_NAME}'. Extend build_selected_pipeline first."
        )
    return model_name


def load_selected_ridge_alpha(search_path: Path) -> float:
    """Return the alpha that the Ridge tuning policy actually refitted."""
    search_path = Path(search_path)
    if not search_path.exists():
        raise FileNotFoundError(f"Ridge alpha search not found: {search_path}")

    candidates = pd.read_csv(search_path)
    selected = candidates.loc[candidates["selected"].astype(bool)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one selected Ridge alpha, found {len(selected)}"
        )
    return float(selected["alpha"].iloc[0])


def evaluate_final_holdout(
    pipeline: Pipeline,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> dict[str, float]:
    """Score the untouched holdout on both the log and the dollar scale."""
    y_pred_log = pipeline.predict(x_holdout)
    y_true_price = np.expm1(y_holdout)
    y_pred_price = np.expm1(y_pred_log)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_holdout, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_holdout, y_pred_log)),
        "r2_log": float(r2_score(y_holdout, y_pred_log)),
        "rmse_price": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "mae_price": float(mean_absolute_error(y_true_price, y_pred_price)),
        "r2_price": float(r2_score(y_true_price, y_pred_price)),
    }


def _payload_value(field_name: str, raw_value: Any) -> Any:
    """Convert one raw cell into a value PredictionInput accepts."""
    if raw_value is None or pd.isna(raw_value):
        return None
    if field_name in INTEGER_PAYLOAD_FIELDS:
        return int(raw_value)
    if isinstance(raw_value, str):
        return raw_value
    return float(raw_value)


def build_demo_samples(
    raw_df: pd.DataFrame,
    x_holdout: pd.DataFrame,
    sample_count: int = DEMO_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Build snake-case demo records from the first holdout rows."""
    samples: list[dict[str, Any]] = []
    for row_index in list(x_holdout.index)[:sample_count]:
        raw_row = raw_df.loc[row_index]
        payload = {
            field_name: _payload_value(field_name, raw_row[column])
            for column, field_name in PAYLOAD_FIELD_BY_COLUMN.items()
        }
        samples.append(
            {
                "index": int(row_index),
                "id": int(raw_row["Id"]) if "Id" in raw_df.columns else None,
                "actual_price": float(raw_row[baseline_round2.TARGET_COLUMN]),
                "payload": payload,
            }
        )
    return samples


def smoke_test_artifact(
    model_path: Path,
    samples: Sequence[dict[str, Any]],
) -> list[float]:
    """Reload the saved artifact and predict a few samples end to end."""
    model, _fingerprint = load_model_artifact(Path(model_path))
    prices: list[float] = []
    for sample in samples:
        price = predict_price(model, PredictionInput(**sample["payload"]))
        if not 0.0 < price < MAX_PLAUSIBLE_PRICE:
            raise ValueError(f"Implausible reloaded prediction: {price}")
        LOGGER.info(
            "Smoke test id=%s predicted=%.2f actual=%.2f",
            sample["id"],
            price,
            sample["actual_price"],
        )
        prices.append(price)
    return prices


def _write_report_readme(artifacts_dir: Path, report: dict[str, Any]) -> None:
    """Write the human-readable companion to the JSON report."""
    metrics = report["metrics"]
    lines = [
        "# Final holdout report",
        "",
        f"- Model: `{report['model_name']}` (alpha={report['ridge_alpha']})",
        f"- Artifact sha256: `{report['model_sha256']}`",
        f"- Train+eval rows: {report['train_eval_rows']}",
        f"- Final holdout rows: {report['holdout_rows']}",
        f"- RMSE log: {metrics['rmse_log']:.6f}",
        f"- MAE log: {metrics['mae_log']:.6f}",
        f"- R2 log: {metrics['r2_log']:.6f}",
        "",
        "## Selection note",
        "",
        SELECTION_NOTE,
        "",
    ]
    (artifacts_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_finalize(
    data_path: Path = DEFAULT_DATA_PATH,
    round3_metrics_path: Path = DEFAULT_ROUND3_METRICS,
    ridge_search_path: Path = DEFAULT_RIDGE_SEARCH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    test_size: float = baseline_round2.DEFAULT_TEST_SIZE,
    random_state: int = baseline_round2.DEFAULT_RANDOM_STATE,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Refit the selected model, save one artifact, and score the holdout once."""
    artifacts_dir = Path(artifacts_dir)
    report_path = artifacts_dir / "final_holdout_report.json"
    if report_path.exists() and not allow_overwrite:
        raise RuntimeError(
            f"Final holdout already scored: {report_path}. The holdout is a "
            "single-use report number; pass allow_overwrite to redo it."
        )

    model_name = load_selected_model_name(round3_metrics_path)
    ridge_alpha = load_selected_ridge_alpha(ridge_search_path)

    x_raw, y_log = baseline_round2.load_raw_training_data(Path(data_path))
    x_train_eval, x_holdout, y_train_eval, y_holdout = (
        baseline_round2.split_raw_holdout(
            x_raw=x_raw,
            y_log=y_log,
            test_size=test_size,
            random_state=random_state,
        )
    )

    pipeline = baseline_round2.build_baseline_pipeline(
        model_name, ridge_alpha=ridge_alpha
    )
    pipeline.fit(x_train_eval, y_train_eval)
    LOGGER.info(
        "Refitted %s (alpha=%s) on %s train+eval rows",
        model_name,
        ridge_alpha,
        len(x_train_eval),
    )

    model_path = Path(model_path)
    atomic_dump(pipeline, model_path)
    _model, model_sha256 = load_model_artifact(model_path)

    metrics = evaluate_final_holdout(pipeline, x_holdout, y_holdout)
    LOGGER.info(
        "Final holdout scored once: rmse_log=%.6f on %s rows",
        metrics["rmse_log"],
        len(x_holdout),
    )

    raw_df = pd.read_csv(Path(data_path))
    samples = build_demo_samples(raw_df=raw_df, x_holdout=x_holdout)

    report: dict[str, Any] = {
        "model_name": model_name,
        "ridge_alpha": ridge_alpha,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "data_path": str(Path(data_path)),
        "test_size": test_size,
        "random_state": random_state,
        "train_eval_rows": len(x_train_eval),
        "holdout_rows": len(x_holdout),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selection_note": SELECTION_NOTE,
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (artifacts_dir / "demo_samples.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    _write_report_readme(artifacts_dir, report)

    report["smoke_test_prices"] = smoke_test_artifact(model_path, samples)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--round3-metrics-path", type=Path, default=DEFAULT_ROUND3_METRICS
    )
    parser.add_argument("--ridge-search-path", type=Path, default=DEFAULT_RIDGE_SEARCH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--test-size", type=float, default=baseline_round2.DEFAULT_TEST_SIZE
    )
    parser.add_argument(
        "--random-state", type=int, default=baseline_round2.DEFAULT_RANDOM_STATE
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run packaging and the one-shot final holdout evaluation."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    return run_finalize(
        data_path=args.data_path,
        round3_metrics_path=args.round3_metrics_path,
        ridge_search_path=args.ridge_search_path,
        artifacts_dir=args.artifacts_dir,
        model_path=args.model_path,
        test_size=args.test_size,
        random_state=args.random_state,
        allow_overwrite=args.allow_overwrite,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
