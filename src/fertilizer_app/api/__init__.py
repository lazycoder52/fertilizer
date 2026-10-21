"""API modules and blueprints."""
from .web import web_bp
from .v1.routes import bp as api_v1_bp

__all__ = ["web_bp", "api_v1_bp"]
