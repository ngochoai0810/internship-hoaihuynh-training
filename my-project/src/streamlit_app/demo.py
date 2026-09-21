"""Pure presentation helpers and fixtures for the technical demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
FINAL_ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "final"
DEMO_SAMPLES_PATH = FINAL_ARTIFACTS_DIR / "demo_samples.json"
FINAL_REPORT_PATH = FINAL_ARTIFACTS_DIR / "final_holdout_report.json"

DEMO_TOTAL_REQUESTS = 1000

QUALITY_OPTIONS: list[str] = ["Po", "Fa", "TA", "Gd", "Ex"]
BASEMENT_QUALITY_OPTIONS: list[str] = ["None", *QUALITY_OPTIONS]
GARAGE_TYPE_OPTIONS: list[str] = [
    "Attchd",
    "Detchd",
    "BuiltIn",
    "Basment",
    "CarPort",
    "2Types",
    "None",
]
NEIGHBORHOOD_OPTIONS: list[str] = [
    "Blmngtn",
    "Blueste",
    "BrDale",
    "BrkSide",
    "ClearCr",
    "CollgCr",
    "Crawfor",
    "Edwards",
    "Gilbert",
    "IDOTRR",
    "MeadowV",
    "Mitchel",
    "NAmes",
    "NPkVill",
    "NWAmes",
    "NoRidge",
    "NridgHt",
    "OldTown",
    "SWISU",
    "Sawyer",
    "SawyerW",
    "Somerst",
    "StoneBr",
    "Timber",
    "Veenker",
]

MANUAL_DEFAULT_PAYLOAD: dict[str, Any] = {
    "overall_qual": 6,
    "gr_liv_area": 1464.0,
    "garage_cars": 2.0,
    "garage_area": 480.0,
    "total_bsmt_sf": 992.0,
    "first_flr_sf": 1087.0,
    "full_bath": 2,
    "tot_rms_abv_grd": 6,
    "year_built": 1973,
    "year_remod_add": 1994,
    "neighborhood": "NAmes",
    "garage_type": "Attchd",
    "exter_qual": "TA",
    "kitchen_qual": "TA",
    "bsmt_qual": "TA",
}

METRIC_LABELS: dict[str, str] = {
    "rmse_log": "RMSE log",
    "mae_log": "MAE log",
    "r2_log": "R²",
}


@dataclass(frozen=True)
class PredictionComparison:
    """Difference between a prediction and its known holdout target."""

    absolute_error: float
    percentage_error: float


@dataclass(frozen=True)
class HoldoutSample:
    """One final holdout row published by `python -m ml.finalize`."""

    sample_id: int | None
    actual_price: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class VerifiedModelEvaluation:
    """Evaluation facts that are valid only for one exact artifact hash."""

    model_sha256: str
    source_dataset: str
    test_size: float
    split_random_state: int
    metrics: dict[str, float]


def _read_json(path: Path) -> Any | None:
    """Return parsed JSON, or None when the artifact has not been produced."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_holdout_sample(path: Path = DEMO_SAMPLES_PATH) -> HoldoutSample | None:
    """Load the first published holdout sample, or None when unavailable."""

    samples = _read_json(path)
    if not samples:
        return None
    first = samples[0]
    return HoldoutSample(
        sample_id=first.get("id"),
        actual_price=float(first["actual_price"]),
        payload=dict(first["payload"]),
    )


def load_verified_evaluation(
    path: Path = FINAL_REPORT_PATH,
) -> VerifiedModelEvaluation | None:
    """Load final holdout facts, or None when the model was never finalized."""

    report = _read_json(path)
    if not report:
        return None
    metrics = report["metrics"]
    return VerifiedModelEvaluation(
        model_sha256=str(report["model_sha256"]),
        source_dataset=str(report["data_path"]),
        test_size=float(report["test_size"]),
        split_random_state=int(report["random_state"]),
        metrics={
            label: float(metrics[key])
            for key, label in METRIC_LABELS.items()
            if key in metrics
        },
    )


HOLDOUT_SAMPLE: HoldoutSample | None = load_holdout_sample()
VERIFIED_MODEL_EVALUATION: VerifiedModelEvaluation | None = load_verified_evaluation()
BENCHMARK_PAYLOAD: dict[str, Any] = (
    HOLDOUT_SAMPLE.payload if HOLDOUT_SAMPLE is not None else MANUAL_DEFAULT_PAYLOAD
)


def get_verified_evaluation(model_sha256: str) -> VerifiedModelEvaluation | None:
    """Return evaluation facts only when they belong to the active artifact."""

    if (
        VERIFIED_MODEL_EVALUATION is not None
        and model_sha256 == VERIFIED_MODEL_EVALUATION.model_sha256
    ):
        return VERIFIED_MODEL_EVALUATION
    return None


def compare_prediction(
    *, predicted_price: float, actual_price: float
) -> PredictionComparison:
    """Calculate per-sample error without presenting it as model accuracy."""

    if actual_price <= 0:
        raise ValueError("Actual price must be greater than zero")
    absolute_error = abs(predicted_price - actual_price)
    return PredictionComparison(
        absolute_error=absolute_error,
        percentage_error=absolute_error / actual_price * 100,
    )


def format_history_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten API history objects into columns suitable for a demo table."""

    rows: list[dict[str, Any]] = []
    for record in records:
        payload = record.get("input_payload", {})
        created_at = str(record.get("created_at", "")).replace("T", " ")
        rows.append(
            {
                "ID": record.get("id"),
                "Thời gian": created_at,
                "Giá dự đoán": f"${float(record.get('predicted_price', 0)):,.2f}",
                "Chất lượng chung": payload.get("overall_qual"),
                "Diện tích ở": payload.get("gr_liv_area"),
                "Tầng hầm": payload.get("total_bsmt_sf"),
                "Khu vực": payload.get("neighborhood"),
                "Garage": payload.get("garage_type"),
                "Model": str(record.get("model_sha256", ""))[:12],
            }
        )
    return rows
