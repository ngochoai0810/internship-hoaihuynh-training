"""Runtime loading and inference helpers for persisted sklearn artifacts."""

import hashlib
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from schemas.prediction import PredictionInput


class PredictionModel(Protocol):
    def predict(self, frame: pd.DataFrame) -> NDArray[np.float64]: ...


def load_model_artifact(path: Path) -> tuple[PredictionModel, str]:
    """Load and fingerprint a prediction artifact or fail startup."""

    model_path = Path(path)
    try:
        artifact_bytes = model_path.read_bytes()
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load model artifact: {model_path}") from exc

    if not callable(getattr(model, "predict", None)):
        raise RuntimeError(f"Invalid model artifact without predict(): {model_path}")

    fingerprint = hashlib.sha256(artifact_bytes).hexdigest()
    return cast(PredictionModel, model), fingerprint


def build_model_frame(payload: PredictionInput) -> pd.DataFrame:
    """Map the public snake-case contract to the fitted pipeline columns."""

    return pd.DataFrame(
        [
            {
                "LotFrontage": payload.lot_frontage,
                "MasVnrArea": payload.mas_vnr_area,
                "TotalBsmtSF": payload.total_bsmt_sf,
                "GarageType": payload.garage_type,
                "Alley": payload.alley,
                "ExterQual": payload.exter_qual,
            }
        ]
    )


def predict_price(model: PredictionModel, payload: PredictionInput) -> float:
    """Run one inference and convert log1p price back to US dollars."""

    prediction = np.asarray(model.predict(build_model_frame(payload)), dtype=float)
    if prediction.shape != (1,) or not np.isfinite(prediction[0]):
        raise ValueError("Model returned an invalid prediction")
    with np.errstate(over="ignore", invalid="ignore"):
        price = float(np.expm1(prediction[0]))
    if not np.isfinite(price) or price < 0:
        raise ValueError("Model returned an invalid price")
    return round(price, 2)
