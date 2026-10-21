# ==============================================================================
# Production Dockerfile for Fertilizer Recommendation ML Application
# Multi-stage, lean, secure (non-root), healthchecked
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation and native library dependencies (OpenMP for LightGBM/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a separate directory for clean copying
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Final Lean Runtime Image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system libraries and healthcheck utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system group and user
RUN addgroup --system --gid 10001 appgroup && \
    adduser --system --uid 10001 --gid 10001 --no-create-home appuser

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy application artifacts and code
COPY --chown=appuser:appgroup app.py pyproject.toml pytest.ini ./
COPY --chown=appuser:appgroup src/ src/
COPY --chown=appuser:appgroup templates/ templates/
COPY --chown=appuser:appgroup static/ static/
COPY --chown=appuser:appgroup model_files/ model_files/
COPY --chown=appuser:appgroup config/ config/

# Security: run as non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    FLASK_ENV=production \
    MODEL_DIR=model_files

EXPOSE 5000

# Docker Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Production WSGI server execution
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "--threads", "2", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
