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


def test_api_v1_info_metrics_match_metrics_json(client):
    """Verify each metric returned by /api/v1/info exactly matches model_files/metrics.json."""
    import json
    from pathlib import Path

    metrics_path = Path("model_files/metrics.json")
    assert metrics_path.exists(), "metrics.json must exist"
    with open(metrics_path, encoding="utf-8") as f:
        metrics_data = json.load(f)

    holdout = metrics_data["holdout_evaluation"]

    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.get_json()

    # Check metrics under model_metadata
    assert "model_metadata" in data
    meta = data["model_metadata"]
    for key, expected_val in holdout.items():
        if key in meta:
            assert meta[key] == pytest.approx(expected_val, abs=1e-6), (
                f"Metric {key} in model_metadata ({meta[key]}) != metrics.json ({expected_val})"
            )

    # Check top-level metrics block
    assert "metrics" in data
    for key, expected_val in holdout.items():
        assert key in data["metrics"]
        assert data["metrics"][key] == pytest.approx(expected_val, abs=1e-6), (
            f"Metric {key} in info metrics ({data['metrics'][key]}) != metrics.json ({expected_val})"
        )


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
    assert data["recommendation"] == "Urea"
    assert data["confidence"] == pytest.approx(86.67, abs=0.5)
    assert "remark" in data
    assert len(data["top_3_recommendations"]) == 3
    assert data["top_3_recommendations"][0]["fertilizer"] == "Urea"
    assert data["top_3_recommendations"][0]["confidence"] == pytest.approx(86.67, abs=0.5)
    assert data["top_3_recommendations"][1]["fertilizer"] == "DAP"
    assert data["top_3_recommendations"][2]["fertilizer"] == "Muriate of Potash"
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
    """Verify Pydantic rejects out-of-bounds moisture (> 100.0 or < 0.0) and converts valid percentages."""
    # Out of bounds rejection
    payload_bad = {"Soil": "Loamy Soil", "Crop": "Wheat", "Moisture": 150.0}
    response_bad = client.post("/api/v1/predict", json=payload_bad)
    assert response_bad.status_code == 400
    data_bad = response_bad.get_json()
    assert "Validation error" in data_bad["error"]

    # Valid percentage (60.0%) accepted and converted to 0.60
    payload_good = {"Soil": "Loamy Soil", "Crop": "Wheat", "Moisture": 60.0}
    response_good = client.post("/api/v1/predict", json=payload_good)
    assert response_good.status_code == 200
    data_good = response_good.get_json()
    assert data_good["input_received"]["Moisture"] == 0.60


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
