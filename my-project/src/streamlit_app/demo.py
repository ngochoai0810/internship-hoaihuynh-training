"""Pure presentation helpers and fixtures for the technical demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HOLDOUT_SAMPLE_ID = 605
HOLDOUT_ACTUAL_PRICE = 221_000.0
VERIFIED_MODEL_SHA256 = (
    "0f3d10f9b4086d6269a2732d9f45821ca5dd3775f0e8e898967a566e1187c7c8"
)
DEMO_TOTAL_REQUESTS = 1000
HOLDOUT_PAYLOAD: dict[str, Any] = {
    "lot_frontage": 88.0,
    "mas_vnr_area": 99.0,
    "total_bsmt_sf": 1494.0,
    "garage_type": "Attchd",
    "alley": None,
    "exter_qual": "Gd",
}


@dataclass(frozen=True)
class PredictionComparison:
    """Difference between a prediction and its known holdout target."""

    absolute_error: float
    percentage_error: float


@dataclass(frozen=True)
class VerifiedModelEvaluation:
    """Evaluation facts that are valid only for one exact artifact hash."""

    model_sha256: str
    sample_id: int
    source_dataset: str
    subset_fraction: float
    subset_random_state: int
    test_size: float
    split_random_state: int
    metrics: dict[str, float]


VERIFIED_MODEL_EVALUATION = VerifiedModelEvaluation(
    model_sha256=VERIFIED_MODEL_SHA256,
    sample_id=HOLDOUT_SAMPLE_ID,
    source_dataset="data/retrain/train_subset.csv",
    subset_fraction=0.6,
    subset_random_state=2026,
    test_size=0.2,
    split_random_state=42,
    metrics={
        "RMSE log": 0.255751,
        "MAE log": 0.187704,
        "R²": 0.631583,
    },
)


def get_verified_evaluation(model_sha256: str) -> VerifiedModelEvaluation | None:
    """Return evaluation facts only when they belong to the active artifact."""

    if model_sha256 == VERIFIED_MODEL_EVALUATION.model_sha256:
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
                "Mặt tiền": payload.get("lot_frontage"),
                "Tầng hầm": payload.get("total_bsmt_sf"),
                "Garage": payload.get("garage_type"),
                "Chất lượng": payload.get("exter_qual"),
                "Model": str(record.get("model_sha256", ""))[:12],
            }
        )
    return rows
