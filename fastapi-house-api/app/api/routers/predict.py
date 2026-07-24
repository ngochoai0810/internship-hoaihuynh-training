"""Mock prediction route for the future house price model."""

import random

from fastapi import APIRouter, HTTPException

from app.schemas.predict import PredictHouseInput, PredictResponse


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_price(house: PredictHouseInput) -> PredictResponse:
    """Return a mock house price prediction using the public prediction contract."""
    if house.gr_liv_area > 20000:
        raise HTTPException(
            status_code=400,
            detail="gr_liv_area exceeds the reasonable maximum of 20000 sqft",
        )

    base_price = (
        house.gr_liv_area * 100
        + house.total_bsmt_sf * 45
        + house.overall_qual * 5000
        + house.garage_cars * 7500
    )
    noise = random.uniform(-0.05, 0.05)
    predicted_price = round(base_price * (1 + noise), 2)

    return PredictResponse(predicted_price=predicted_price)
