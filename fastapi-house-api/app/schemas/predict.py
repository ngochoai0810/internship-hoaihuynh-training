"""Schemas for the mock house price prediction endpoint."""

from pydantic import BaseModel, Field


class PredictHouseInput(BaseModel):
    """Input contract for house price prediction."""

    gr_liv_area: float = Field(..., gt=0, description="Living area above ground in sqft")
    overall_qual: int = Field(..., ge=1, le=10, description="Overall quality score from 1 to 10")
    year_built: int = Field(..., ge=1800, le=2026, description="Year the house was built")
    total_bsmt_sf: float = Field(0, ge=0, description="Total basement area in sqft")
    garage_cars: int = Field(0, ge=0, le=10, description="Garage capacity in cars")


class PredictResponse(BaseModel):
    """Stable output contract for mock and future real predictions."""

    predicted_price: float = Field(..., description="Predicted house price")
    currency: str = Field(default="USD", description="Prediction currency")
