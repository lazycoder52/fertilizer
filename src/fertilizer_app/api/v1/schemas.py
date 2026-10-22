"""Pydantic v2 Validation Schemas for Fertilizer Recommendation API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Input payload for fertilizer prediction with realistic agronomic bounds."""

    Soil: str = Field(..., description="Soil type (e.g. Loamy Soil, Peaty Soil, Acidic Soil)")
    Crop: str = Field(..., description="Crop type (e.g. Wheat, Rice, Apple, Cotton)")

    Temperature: float | None = Field(
        default=None,
        ge=-20.0,
        le=60.0,
        description="Ambient temperature in degrees Celsius (-20 to 60)",
    )
    Moisture: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Soil moisture ratio (0.0 to 1.0)",
    )
    Rainfall: float | None = Field(
        default=None,
        ge=0.0,
        le=1000.0,
        description="Rainfall in millimeters (0 to 1000)",
    )
    PH: float | None = Field(
        default=None,
        ge=0.0,
        le=14.0,
        description="Soil pH level (0.0 to 14.0)",
    )
    Nitrogen: float | None = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Soil Nitrogen content (0 to 500 kg/ha)",
    )
    Phosphorous: float | None = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Soil Phosphorus content (0 to 500 kg/ha)",
    )
    Potassium: float | None = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Soil Potassium content (0 to 500 kg/ha)",
    )
    Carbon: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Soil organic carbon percentage (0 to 10%)",
    )

    @field_validator("Soil", "Crop")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class RecommendationItem(BaseModel):
    """Single ranked fertilizer recommendation."""

    fertilizer: str
    confidence: float = Field(..., description="Confidence score percentage (0-100)")
    remark: str = Field(default="", description="Agronomic rationale and usage guidance")


class PredictResponse(BaseModel):
    """Output payload for prediction endpoint."""

    success: bool = True
    recommendation: str
    confidence: float
    remark: str
    top_3_recommendations: list[RecommendationItem]
    input_received: dict[str, Any]
    is_imputed: bool = Field(
        default=False,
        description="True if any soil test parameters were omitted and imputed with benchmark medians",
    )
    imputed_fields: list[str] = Field(
        default_factory=list,
        description="List of omitted features that were imputed with benchmark medians",
    )
    warning: str | None = Field(
        default=None,
        description="Agronomic trust warning displayed when default imputed values are used",
    )
    explanation: dict[str, Any] | None = Field(
        default=None,
        description="SHAP local feature attribution breakdown",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
