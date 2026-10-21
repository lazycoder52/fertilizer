"""Version 1 API Routes Blueprint."""

import logging
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from src.fertilizer_app.api.v1.schemas import PredictRequest
from src.fertilizer_app.core.limiter import limiter
from src.fertilizer_app.services.model_service import ModelService

logger = logging.getLogger(__name__)

bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


@bp.route("/predict", methods=["POST"])
@limiter.limit("60 per minute")
def predict_v1():
    """Fertilizer Recommendation Prediction Endpoint.
    ---
    tags:
      - Fertilizer Recommendation
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - Soil
            - Crop
          properties:
            Soil:
              type: string
              example: Loamy Soil
            Crop:
              type: string
              example: Wheat
            Nitrogen:
              type: number
              example: 45.0
            Phosphorous:
              type: number
              example: 30.0
            Potassium:
              type: number
              example: 20.0
            PH:
              type: number
              example: 6.5
            Moisture:
              type: number
              example: 0.65
            Temperature:
              type: number
              example: 26.0
            Rainfall:
              type: number
              example: 120.0
            Carbon:
              type: number
              example: 1.5
    responses:
      200:
        description: Successful prediction with top 3 ranked recommendations
      400:
        description: Validation error or invalid category
      429:
        description: Rate limit exceeded
    """
    raw_json = request.get_json(silent=True)
    if raw_json is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # 1. Pydantic validation
    try:
        validated_request = PredictRequest.model_validate(raw_json)
    except ValidationError as err:
        errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in err.errors()
        ]
        return jsonify({"error": "Validation error", "details": errors}), 400

    # 2. Model inference via Singleton service
    service = ModelService()
    try:
        result = service.predict(validated_request.model_dump())
        return jsonify(result), 200
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as exc:
        logger.exception("Unexpected prediction error: %s", exc)
        return jsonify({"error": "Internal inference error", "details": str(exc)}), 500


@bp.route("/info", methods=["GET"])
def info_v1():
    """Retrieve Model Information and Valid Categorical Options.
    ---
    tags:
      - System Info
    responses:
      200:
        description: Model metadata, features, and supported classes
    """
    service = ModelService()
    return jsonify(service.get_info()), 200


@bp.route("/health", methods=["GET"])
def health_v1():
    """Health check probe.
    ---
    tags:
      - System Info
    responses:
      200:
        description: Service is healthy
    """
    return jsonify({"status": "healthy"}), 200


@bp.route("/metrics", methods=["GET"])
def metrics_v1():
    """Retrieve Operational Telemetry and Prediction Counters.
    ---
    tags:
      - System Info
    responses:
      200:
        description: Aggregated telemetry counters
    """
    service = ModelService()
    return jsonify(service.get_metrics()), 200
