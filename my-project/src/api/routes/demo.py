"""Authenticated access to artifacts produced with the active model."""

import json
from pathlib import Path
from typing import Annotated, Any

from dependencies.auth import get_current_user
from fastapi import APIRouter, Depends, Request
from models import User
from pydantic import ValidationError
from schemas.demo import (
    DemoArtifactsResponse,
    DemoEvaluationResponse,
    DemoSampleResponse,
)

router = APIRouter(tags=["demo"])

DISPLAY_METRICS = ("rmse_log", "mae_log", "r2_log")


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_sample(path: Path) -> DemoSampleResponse | None:
    samples = _read_json(path)
    if not isinstance(samples, list) or not samples:
        return None
    try:
        return DemoSampleResponse.model_validate(samples[0])
    except ValidationError:
        return None


def _load_evaluation(
    path: Path,
    *,
    active_model_sha256: str,
) -> DemoEvaluationResponse | None:
    report = _read_json(path)
    if not isinstance(report, dict):
        return None
    if report.get("model_sha256") != active_model_sha256:
        return None
    try:
        metrics = report["metrics"]
        return DemoEvaluationResponse(
            model_sha256=report["model_sha256"],
            source_dataset=report["data_path"],
            test_size=report["test_size"],
            split_random_state=report["random_state"],
            metrics={name: metrics[name] for name in DISPLAY_METRICS},
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


@router.get("/demo", response_model=DemoArtifactsResponse)
def read_demo_artifacts(
    request: Request,
    _current_user: Annotated[User, Depends(get_current_user)],
) -> DemoArtifactsResponse:
    """Return a demo sample and only metrics verified for the active model."""

    artifacts_dir = request.app.state.settings.final_artifacts_dir
    model_sha256 = request.app.state.model_sha256
    return DemoArtifactsResponse(
        model_sha256=model_sha256,
        sample=_load_sample(artifacts_dir / "demo_samples.json"),
        evaluation=_load_evaluation(
            artifacts_dir / "final_holdout_report.json",
            active_model_sha256=model_sha256,
        ),
    )
