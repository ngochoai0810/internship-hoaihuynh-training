"""Contracts for model-matched demo artifacts."""

from pydantic import BaseModel, Field
from schemas.prediction import PredictionInput


class DemoSampleResponse(BaseModel):
    id: int | None
    actual_price: float = Field(gt=0)
    payload: PredictionInput


class DemoEvaluationResponse(BaseModel):
    model_sha256: str
    source_dataset: str
    test_size: float = Field(gt=0, lt=1)
    split_random_state: int
    metrics: dict[str, float]


class DemoArtifactsResponse(BaseModel):
    model_sha256: str
    sample: DemoSampleResponse | None
    evaluation: DemoEvaluationResponse | None
