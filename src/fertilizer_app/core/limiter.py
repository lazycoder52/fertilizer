"""Rate Limiter Initialization."""

import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

ratelimit_enabled = os.environ.get("RATELIMIT_ENABLED", "true").lower() in ("true", "1")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120 per minute"],
    storage_uri="memory://",
    enabled=ratelimit_enabled,
)
