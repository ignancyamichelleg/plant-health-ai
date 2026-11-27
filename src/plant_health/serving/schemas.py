"""Pydantic response schemas for the PlantHealth AI API."""

from typing import Literal

from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    """Image quality measurements."""

    brightness: float = Field(ge=0, le=255)
    blur: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Response returned by POST /predict."""

    request_id: str
    status: Literal["ok", "uncertain", "low_quality"]
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    quality: QualityMetrics
    model_version: str
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: Literal["ok"]
    model_version: str
