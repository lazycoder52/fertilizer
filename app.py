"""Top-level entrypoint for Fertilizer Recommendation System.

Initializes the application using the factory pattern and runs the dev server.
For production deployments, Gunicorn binds to `app:app`.
"""

import os
import sys

# Support disabling rate limiter for local load testing and benchmarking
if "--no-limiter" in sys.argv:
    os.environ["RATELIMIT_ENABLED"] = "0"

from src.fertilizer_app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
