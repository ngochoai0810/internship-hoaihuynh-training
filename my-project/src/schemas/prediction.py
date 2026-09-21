"""Live house-price prediction contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    overall_qual: int = Field(ge=0)
    gr_liv_area: float = Field(ge=0)
    garage_cars: float | None = Field(default=None, ge=0)
    garage_area: float | None = Field(default=None, ge=0)
    total_bsmt_sf: float | None = Field(default=None, ge=0)
    first_flr_sf: float = Field(ge=0)
    full_bath: int = Field(ge=0)
    tot_rms_abv_grd: int = Field(ge=0)
    year_built: int = Field(ge=0)
    year_remod_add: int = Field(ge=0)
    neighborhood: str = Field(min_length=1)
    garage_type: str | None = Field(default=None, min_length=1)
    exter_qual: Literal["Po", "Fa", "TA", "Gd", "Ex"] | None = None
    kitchen_qual: Literal["Po", "Fa", "TA", "Gd", "Ex"] | None = None
    bsmt_qual: Literal["None", "Po", "Fa", "TA", "Gd", "Ex"] | None = None


class PredictionResponse(BaseModel):
    prediction_history_id: int
    predicted_price: float
    currency: str = "USD"
    model_sha256: str


class PredictionHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    input_payload: dict[str, object]
    predicted_price: float
    model_sha256: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_sha256: str
