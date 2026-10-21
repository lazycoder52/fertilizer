"""Web UI and Legacy Compatibility Blueprint."""

from flask import Blueprint, jsonify, render_template, request

from src.fertilizer_app.services.model_service import ModelService

web_bp = Blueprint("web", __name__)


@web_bp.route("/")
def index():
    """Render the Frontend User Interface."""
    service = ModelService()
    return render_template(
        "index.html",
        soil_types=service.soil_types,
        crop_types=service.crop_types,
    )


@web_bp.route("/predict", methods=["POST"])
def legacy_predict():
    """Backward-compatible /predict endpoint forwarding to service."""
    raw_json = request.get_json(silent=True)
    if raw_json is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    service = ModelService()
    try:
        result = service.predict(raw_json)
        return jsonify(result), 200
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@web_bp.route("/api/info", methods=["GET"])
def legacy_info():
    """Backward-compatible /api/info endpoint."""
    service = ModelService()
    return jsonify(service.get_info()), 200


@web_bp.route("/health", methods=["GET"])
def root_health():
    """Root health check probe for platform load balancers."""
    return jsonify({"status": "healthy"}), 200
