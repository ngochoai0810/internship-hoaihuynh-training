"""Authenticated live model inference route."""

from typing import Annotated

from database import get_db
from dependencies.auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request, status
from ml.runtime import predict_price
from models import PredictionHistory, User
from schemas.prediction import PredictionInput, PredictionResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_prediction(
    payload: PredictionInput,
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    """Run authenticated inference and persist the completed transaction."""

    try:
        price = predict_price(request.app.state.model, payload)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed",
        ) from exc

    history = PredictionHistory(
        user_id=current_user.id,
        input_payload=payload.model_dump(mode="json"),
        predicted_price=price,
        model_sha256=request.app.state.model_sha256,
    )
    db.add(history)
    try:
        db.commit()
        db.refresh(history)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save prediction history",
        ) from exc

    return PredictionResponse(
        prediction_history_id=history.id,
        predicted_price=price,
        model_sha256=request.app.state.model_sha256,
    )
