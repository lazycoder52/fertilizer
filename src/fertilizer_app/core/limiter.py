from flask import current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.fertilizer_app.core.config import config


def get_default_limit() -> str:
    """Resolve default rate limit from Flask app config, AppConfig, or environment."""
    if current_app:
        return current_app.config.get("RATE_LIMIT_DEFAULT", config.rate_limit_default)
    return config.rate_limit_default


def get_predict_limit() -> str:
    """Resolve predict route rate limit from Flask app config, AppConfig, or environment."""
    if current_app:
        return current_app.config.get("RATE_LIMIT_PREDICT", config.rate_limit_predict)
    return config.rate_limit_predict


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[get_default_limit],
    storage_uri="memory://",
    enabled=config.ratelimit_enabled,
)
