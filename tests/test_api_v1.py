"""Integration and Unit Tests for Fertilizer Recommendation Backend."""

import pytest

from src.fertilizer_app import create_app


@pytest.fixture
def client():
    """Create test client with testing config."""
    app = create_app({"TESTING": True, "RATELIMIT_ENABLED": False})
    with app.test_client() as test_client:
        yield test_client


def test_web_index(client):
    """Verify web root renders frontend template."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Fertilizer Recommendation" in response.data


def test_api_v1_health(client):
    """Verify /api/v1/health returns healthy."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


def test_api_v1_info(client):
    """Verify /api/v1/info returns metadata and categories."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.get_json()
    assert "soil_types" in data
    assert "crop_types" in data
    assert "fertilizer_classes" in data
    assert "model_metadata" in data
    assert len(data["soil_types"]) > 0


def test_api_v1_metrics(client):
    """Verify /api/v1/metrics returns operational counters."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "counters" in data
    assert data["status"] == "healthy"


def test_api_v1_predict_valid(client):
    """Verify /api/v1/predict returns recommendation and top 3 ranked items."""
    payload = {
        "Soil": "Loamy Soil",
        "Crop": "Wheat",
        "Nitrogen": 40.0,
        "Phosphorous": 25.0,
        "Potassium": 15.0,
        "PH": 6.5,
        "Moisture": 0.60,
        "Temperature": 24.0,
        "Rainfall": 120.0,
        "Carbon": 1.2,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "recommendation" in data
    assert "confidence" in data
    assert "remark" in data
    assert len(data["top_3_recommendations"]) == 3
    assert data["top_3_recommendations"][0]["fertilizer"] == data["recommendation"]
    assert "explanation" in data
    assert data["explanation"]["enabled"] is True
    assert len(data["explanation"]["top_contributions"]) > 0
    assert data["is_imputed"] is False
    assert data["warning"] is None


def test_api_v1_predict_partial_imputation(client):
    """Verify partial input returns is_imputed=True with visible low-confidence warning."""
    payload = {"Soil": "Loamy Soil", "Crop": "Wheat"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["is_imputed"] is True
    assert len(data["imputed_fields"]) == 8
    assert data["warning"] is not None
    assert "Low confidence" in data["warning"]


def test_api_v1_explain_valid(client):
    """Verify /api/v1/explain returns detailed SHAP explanation payload."""
    payload = {
        "Soil": "Loamy Soil",
        "Crop": "Wheat",
        "Nitrogen": 40.0,
        "Phosphorous": 25.0,
        "Potassium": 15.0,
    }
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "explanation" in data
    assert data["explanation"]["enabled"] is True
    assert len(data["explanation"]["top_contributions"]) > 0
    assert "positive_drivers" in data["explanation"]


def test_api_v1_predict_pydantic_bounds_ph(client):
    """Verify Pydantic rejects out-of-bounds pH (> 14.0)."""
    payload = {"Soil": "Loamy Soil", "Crop": "Wheat", "PH": 25.0}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Validation error" in data["error"]
    assert any("PH" in str(d["loc"]) for d in data["details"])


def test_api_v1_predict_pydantic_bounds_moisture(client):
    """Verify Pydantic rejects out-of-bounds moisture (> 1.0)."""
    payload = {"Soil": "Loamy Soil", "Crop": "Wheat", "Moisture": 5.0}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Validation error" in data["error"]


def test_api_v1_predict_missing_required(client):
    """Verify Pydantic rejects request missing required Soil field."""
    payload = {"Crop": "Wheat"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Validation error" in data["error"]


def test_api_v1_predict_invalid_category(client):
    """Verify rejection of non-existent soil category."""
    payload = {"Soil": "Moon Soil", "Crop": "Wheat"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid soil type" in response.get_json()["error"]


def test_legacy_endpoints(client):
    """Verify legacy root endpoints for backward compatibility."""
    # Legacy health
    assert client.get("/health").status_code == 200
    # Legacy info
    assert client.get("/api/info").status_code == 200
    # Legacy predict
    payload = {"Soil": "Loamy", "Crop": "Wheat"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.get_json()["success"] is True


def test_swagger_docs(client):
    """Verify OpenAPI Swagger documentation UI is accessible."""
    response = client.get("/apidocs/")
    assert response.status_code == 200
