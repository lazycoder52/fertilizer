"""Fertilizer Recommendation Application Factory."""

from pathlib import Path
from typing import Any

from flasgger import Swagger
from flask import Flask

from src.fertilizer_app.api.v1.routes import bp as api_v1_bp
from src.fertilizer_app.api.web import web_bp
from src.fertilizer_app.core.config import config
from src.fertilizer_app.core.limiter import limiter
from src.fertilizer_app.core.logging import init_logging
from src.fertilizer_app.services.model_service import ModelService


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create, configure, and return the Flask application instance."""
    base_dir = Path(__file__).resolve().parents[2]
    template_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    from src.fertilizer_app.core.config import DEFAULT_DEV_SECRET_KEY

    # Core Configurations
    app.config["SECRET_KEY"] = config.secret_key
    app.config["RATE_LIMIT_DEFAULT"] = config.rate_limit_default
    app.config["RATE_LIMIT_PREDICT"] = config.rate_limit_predict
    app.config["RATELIMIT_ENABLED"] = (
        config.ratelimit_enabled
        if hasattr(config, "ratelimit_enabled")
        else __import__("os").environ.get("RATELIMIT_ENABLED", "true").lower() in ("true", "1")
    )
    app.config["SWAGGER"] = {
        "title": "Fertilizer Recommendation API",
        "uiversion": 3,
        "description": "Production ML API providing fertilizer recommendations based on soil and crop conditions.",
        "version": "1.0.0",
        "specs_route": "/apidocs/",
    }

    if test_config:
        app.config.update(test_config)

    # Initialize Logging & Middleware
    init_logging(app)

    # Warn if insecure default secret key is in use
    if app.config.get("SECRET_KEY") == DEFAULT_DEV_SECRET_KEY:
        app.logger.warning(
            "Security Warning: Default development SECRET_KEY is in use! "
            "Set SECRET_KEY in environment for production deployments."
        )

    limiter.init_app(app)

    # Initialize Swagger UI
    if config.swagger_enabled:
        Swagger(app)

    # Register Blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_v1_bp)

    # Pre-warm Model Service (load once on startup)
    with app.app_context():
        ModelService(model_dir=config.model_dir)

    return app
