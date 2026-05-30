# syntax=docker/dockerfile:1
# =============================================================================
# AUDI — Type A Docker Image
#
# Targets Raspberry Pi (arm64/aarch64). Multi-stage build:
#   1. base     — Python + system deps
#   2. runtime  — minimal runtime image
#
# Build:
#   docker build -t audi:latest -f docker/Dockerfile .
#   docker compose -f docker/docker-compose.yml up
# =============================================================================

# --- Stage 1: Base ----------------------------------------------------------
ARG TARGETPLATFORM
FROM --platform=${TARGETPLATFORM} python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps: audio, compression, GPIO, build tools (for lgpio C ext)
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    flac \
    libasound2-dev \
    build-essential \
    python3-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential python3-dev swig && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 2: Runtime -------------------------------------------------------
FROM base AS runtime

# Create data directories
RUN mkdir -p /data/recordings /var/log

# Copy application
COPY src/ /app/src/
COPY models/ /app/models/
COPY webui/ /app/webui/
COPY config.yaml /app/config.yaml
COPY threshold_profiles.yaml /app/threshold_profiles.yaml
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Create a non-root user for the app
RUN useradd -m -u 1000 -s /bin/bash audio-guard && \
    chown -R audio-guard:audio-guard /app /data /var/log

USER audio-guard
WORKDIR /app

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/status')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
