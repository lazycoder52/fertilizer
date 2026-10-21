"""Application Configuration Module."""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application settings with environment variable overrides."""

    env: str = Field(default_factory=lambda: os.getenv("FLASK_ENV", "production"))
    debug: bool = Field(
        default_factory=lambda: os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true")
    )
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "fertilizer-default-secret-key")
    )

    # Paths
    base_dir: Path = Path(__file__).resolve().parents[3]
    model_dir: Path = Field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "model_files")))

    # Rate Limiting
    rate_limit_default: str = Field(
        default_factory=lambda: os.getenv("RATE_LIMIT_DEFAULT", "120 per minute")
    )
    rate_limit_predict: str = Field(
        default_factory=lambda: os.getenv("RATE_LIMIT_PREDICT", "60 per minute")
    )

    # Swagger
    swagger_enabled: bool = True


config = AppConfig()
