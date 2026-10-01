"""schemas.py — Pydantic request/response models for the FastAPI app."""

from typing import Literal
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    study_hours: float = Field(..., ge=0, le=24, description="Study hours per day")
    sleep_hours: float = Field(..., ge=0, le=24, description="Sleep hours per day")
    model: Literal["linear", "random_forest"] = "linear"


class PredictResponse(BaseModel):
    predicted_score: float
    lower_bound: float
    upper_bound: float
    model_used: str
