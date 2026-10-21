"""Locust Load Testing Scenario for Fertilizer Recommendation API.

Simulates concurrent agronomic user requests, including:
- Web UI visits
- Health & Model Info queries
- Full ML inference with all sensor values
- Partial ML inference requiring feature imputation
- On-demand TreeSHAP explainability queries
"""

import random

from locust import HttpUser, between, task


class FertilizerUser(HttpUser):
    """Simulates agricultural field workers and systems submitting fertilizer queries."""

    wait_time = between(1, 2)

    SOILS = ["Acidic Soil", "Alkaline Soil", "Loamy Soil", "Neutral Soil", "Peaty Soil"]
    CROPS = ["wheat", "rice", "maize", "apple", "banana", "grapes", "Cotton", "Coffee"]

    @task(2)
    def visit_index(self):
        """Simulate loading the frontend interface."""
        self.client.get("/", name="[UI] Index Page")

    @task(1)
    def check_health_and_info(self):
        """Simulate operational healthcheck probes and metadata queries."""
        self.client.get("/api/v1/health", name="[API] Health Check")
        self.client.get("/api/v1/info", name="[API] Model Info")

    @task(5)
    def predict_full_payload(self):
        """Simulate full agronomic sensor payload prediction."""
        payload = {
            "Soil": random.choice(self.SOILS),
            "Crop": random.choice(self.CROPS),
            "Temperature": round(random.uniform(15.0, 38.0), 2),
            "Moisture": round(random.uniform(0.15, 0.85), 2),
            "Rainfall": round(random.uniform(20.0, 300.0), 2),
            "PH": round(random.uniform(5.0, 8.5), 2),
            "Nitrogen": round(random.uniform(10.0, 150.0), 2),
            "Phosphorous": round(random.uniform(5.0, 100.0), 2),
            "Potassium": round(random.uniform(5.0, 80.0), 2),
            "Carbon": round(random.uniform(0.2, 3.5), 2),
        }
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            name="[API] Predict (Full Sensors)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200 and resp.json().get("success"):
                resp.success()
            else:
                resp.failure(f"Unexpected response: {resp.status_code} - {resp.text}")

    @task(3)
    def predict_partial_payload(self):
        """Simulate mobile user query with only basic soil and crop selection."""
        payload = {
            "Soil": random.choice(self.SOILS),
            "Crop": random.choice(self.CROPS),
            "Nitrogen": round(random.uniform(10.0, 120.0), 2),
        }
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            name="[API] Predict (Partial / Imputed)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200 and resp.json().get("success"):
                resp.success()
            else:
                resp.failure(f"Unexpected response: {resp.status_code} - {resp.text}")

    @task(2)
    def explain_recommendation(self):
        """Simulate user opening the 'Why this?' SHAP explanation panel."""
        payload = {
            "Soil": random.choice(self.SOILS),
            "Crop": random.choice(self.CROPS),
            "Nitrogen": round(random.uniform(20.0, 100.0), 2),
            "PH": round(random.uniform(5.5, 7.5), 2),
        }
        with self.client.post(
            "/api/v1/explain", json=payload, name="[API] Explain (SHAP)", catch_response=True
        ) as resp:
            if resp.status_code == 200 and resp.json().get("success"):
                resp.success()
            else:
                resp.failure(f"Unexpected response: {resp.status_code} - {resp.text}")
