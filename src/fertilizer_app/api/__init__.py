"""API modules and blueprints."""

from .v1.routes import bp as api_v1_bp
from .web import web_bp

__all__ = ["web_bp", "api_v1_bp"]
