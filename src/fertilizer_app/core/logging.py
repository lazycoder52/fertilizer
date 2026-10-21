"""Structured JSON Logging Module."""

import json
import logging
import time
import uuid
from flask import Flask, g, request


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include request context if available
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "client_ip"):
            log_obj["client_ip"] = record.client_ip
        if hasattr(record, "endpoint"):
            log_obj["endpoint"] = record.endpoint

        return json.dumps(log_obj)


def init_logging(app: Flask) -> None:
    """Initialize structured logging and request telemetry hooks on the Flask app."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    # Configure root and app logger
    app.logger.handlers = [handler]
    app.logger.setLevel(logging.INFO)

    @app.before_request
    def before_request() -> None:
        g.start_time = time.time()
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

    @app.after_request
    def after_request(response):
        # Exclude static assets from access log flooding
        if not request.path.startswith("/static"):
            latency_ms = round((time.time() - getattr(g, "start_time", time.time())) * 1000, 2)
            log_data = {
                "request_id": getattr(g, "request_id", "-"),
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "client_ip": request.remote_addr,
            }
            app.logger.info("request_completed: %s", json.dumps(log_data))
            response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        return response
