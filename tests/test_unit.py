"""Unit Tests for Fertilizer Recommendation Schemas, Config, Logging, and Services."""

import logging

import pytest
from pydantic import ValidationError

from src.fertilizer_app.api.v1.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RecommendationItem,
)
from src.fertilizer_app.core.config import AppConfig, config
from src.fertilizer_app.core.logging import JSONFormatter
from src.fertilizer_app.services.model_service import ModelService


class TestSchemas:
    """Test suite for Pydantic v2 schemas."""

    def test_valid_full_request(self):
        """Test full payload with all soil test values provided."""
        payload = {
            "Soil": "Loamy Soil",
            "Crop": "Wheat",
            "Temperature": 25.5,
            "Moisture": 0.45,
            "Rainfall": 120.0,
            "PH": 6.8,
            "Nitrogen": 80.0,
            "Phosphorous": 40.0,
            "Potassium": 45.0,
            "Carbon": 0.8,
        }
        req = PredictRequest(**payload)
        assert req.Soil == "Loamy Soil"
        assert req.Crop == "Wheat"
        assert req.Temperature == 25.5
        assert req.Moisture == 0.45

    def test_valid_partial_request(self):
        """Test partial payload with only required categorical features."""
        payload = {
            "Soil": "Clay Soil",
            "Crop": "Rice",
        }
        req = PredictRequest(**payload)
        assert req.Soil == "Clay Soil"
        assert req.Crop == "Rice"
        assert req.Temperature is None
        assert req.Nitrogen is None

    def test_empty_soil_or_crop_validation(self):
        """Test validation error when Soil or Crop is whitespace or empty."""
        with pytest.raises(ValidationError):
            PredictRequest(Soil="   ", Crop="Wheat")

        with pytest.raises(ValidationError):
            PredictRequest(Soil="Clay", Crop="")

    def test_ph_out_of_bounds(self):
        """Test validation error when pH exceeds physical bounds."""
        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", PH=15.0)

        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", PH=-1.0)

    def test_moisture_out_of_bounds(self):
        """Test validation error when moisture exceeds [0.0, 100.0]."""
        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Moisture=150.0)

        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Moisture=-0.1)

    def test_moisture_percentage_auto_conversion(self):
        """Test automatic conversion of percentage inputs (> 1.0) to fractions [0.0, 1.0]."""
        req_pct = PredictRequest(Soil="Loamy", Crop="Wheat", Moisture=60.0)
        assert req_pct.Moisture == 0.60

        req_frac = PredictRequest(Soil="Loamy", Crop="Wheat", Moisture=0.60)
        assert req_frac.Moisture == 0.60

        req_full = PredictRequest(Soil="Loamy", Crop="Wheat", Moisture=1.0)
        assert req_full.Moisture == 1.0

    def test_temperature_out_of_bounds(self):
        """Test validation error when temperature exceeds [-20, 60]."""
        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Temperature=75.0)

        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Temperature=-30.0)

    def test_rainfall_out_of_bounds(self):
        """Test validation error when rainfall is negative or above 1000."""
        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Rainfall=-5.0)

        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Rainfall=1500.0)

    def test_carbon_out_of_bounds(self):
        """Test validation error when carbon exceeds 10%."""
        with pytest.raises(ValidationError):
            PredictRequest(Soil="Loamy", Crop="Wheat", Carbon=15.0)

    def test_recommendation_item_schema(self):
        """Test valid serialization of RecommendationItem."""
        item = RecommendationItem(fertilizer="Urea", confidence=98.5, remark="High Nitrogen")
        assert item.fertilizer == "Urea"
        assert item.confidence == 98.5
        assert item.remark == "High Nitrogen"

    def test_predict_response_schema(self):
        """Test valid serialization of PredictResponse."""
        resp = PredictResponse(
            success=True,
            recommendation="Urea",
            confidence=98.5,
            remark="High Nitrogen for leafy growth",
            top_3_recommendations=[
                RecommendationItem(fertilizer="Urea", confidence=98.5),
                RecommendationItem(fertilizer="DAP", confidence=1.0),
                RecommendationItem(fertilizer="Compost", confidence=0.5),
            ],
            input_received={"Soil": "Loamy", "Crop": "Wheat"},
        )
        assert resp.recommendation == "Urea"
        assert len(resp.top_3_recommendations) == 3

    def test_health_response_schema(self):
        """Test HealthResponse schema."""
        h = HealthResponse()
        assert h.status == "healthy"


class TestConfig:
    """Test suite for application environment configuration."""

    def test_config_defaults(self):
        assert config.base_dir.exists()
        assert config.rate_limit_predict == "60 per minute"
        assert config.rate_limit_default == "120 per minute"

    def test_custom_app_config(self):
        custom_cfg = AppConfig(env="testing", debug=True, rate_limit_predict="10 per minute")
        assert custom_cfg.env == "testing"
        assert custom_cfg.debug is True
        assert custom_cfg.rate_limit_predict == "10 per minute"


class TestLogging:
    """Test suite for structured JSON logging."""

    def test_request_log_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="User prediction requested",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "timestamp" in output
        assert "level" in output
        assert "User prediction requested" in output


class TestServices:
    """Test suite for ModelService and ShapService."""

    def test_model_service_load_and_info(self):
        service = ModelService()
        info = service.get_info()
        assert "soil_types" in info
        assert "crop_types" in info
        assert "fertilizer_classes" in info
        assert len(info["fertilizer_classes"]) == 10
        assert len(service.fertilizer_classes) == 10
        assert len(service.required_features) == 10

    def test_model_service_predict_full(self):
        service = ModelService()
        res = service.predict(
            {
                "Soil": "Loamy Soil",
                "Crop": "wheat",
                "Temperature": 30.0,
                "Moisture": 0.60,
                "Rainfall": 150.0,
                "PH": 6.5,
                "Nitrogen": 45.0,
                "Phosphorous": 55.0,
                "Potassium": 40.0,
                "Carbon": 0.5,
            }
        )
        assert res["success"] is True
        assert "recommendation" in res
        assert "confidence" in res
        assert res["confidence"] > 0
        assert res["is_imputed"] is False
        assert len(res["imputed_fields"]) == 0
        assert res["warning"] is None
        assert len(res["top_3_recommendations"]) == 3
        assert "explanation" in res
        assert res["explanation"]["enabled"] is True

    def test_model_service_predict_partial_imputation(self):
        service = ModelService()
        res = service.predict(
            {
                "Soil": "Peaty Soil",
                "Crop": "rice",
            }
        )
        assert res["success"] is True
        assert "recommendation" in res
        assert res["is_imputed"] is True
        assert len(res["imputed_fields"]) == 8
        assert res["warning"] is not None
        assert "Low confidence" in res["warning"]
        assert "explanation" in res

    def test_model_service_invalid_soil_crop(self):
        service = ModelService()
        with pytest.raises(ValueError) as exc:
            service.predict({"Soil": "NonExistentSoil", "Crop": "wheat"})
        assert "Invalid soil type" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            service.predict({"Soil": "Loamy Soil", "Crop": "NonExistentCrop"})
        assert "Invalid crop type" in str(exc.value)

    def test_shap_service_explain(self):
        service = ModelService()
        assert service.shap_service.explainer is not None

        input_data = {
            "Soil": "Loamy Soil",
            "Crop": "wheat",
            "Temperature": 30.0,
            "Moisture": 0.60,
            "Rainfall": 150.0,
            "PH": 6.5,
            "Nitrogen": 45.0,
            "Phosphorous": 55.0,
            "Potassium": 40.0,
            "Carbon": 0.5,
        }
        res = service.predict(input_data)
        explanation = res["explanation"]

        assert explanation is not None
        assert explanation["enabled"] is True
        assert "top_contributions" in explanation
        assert 0 < len(explanation["top_contributions"]) <= 6
        assert "positive_drivers" in explanation
        assert "negative_drivers" in explanation
