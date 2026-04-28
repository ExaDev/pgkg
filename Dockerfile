# syntax=docker/dockerfile:1

# Stage 1: builder — install dependencies via uv
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./
# Copy uv.lock if present (tolerate absence with a conditional COPY)
COPY uv.lock* ./

# Sync production dependencies only (no dev extras)
# CPU-only torch is acceptable here; no GPU support in this base image.
# sentence-transformers will pull in a large torch wheel — this is expected.
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Copy source code (pgkg/ may not exist yet if Phase 2a is still in progress)
COPY pgkg/ ./pgkg/
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/


# Stage 2: runtime — lean image with venv + app code only
FROM python:3.12-slim AS runtime

# Create non-root user
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code and migrations
COPY --from=builder /app/pgkg ./pgkg
COPY --from=builder /app/migrations ./migrations
COPY --from=builder /app/scripts ./scripts

# Ensure scripts are executable
RUN chmod +x ./scripts/*.sh 2>/dev/null || true

# Give appuser ownership
RUN chown -R appuser:appuser /app

# HuggingFace model cache — mounted as a named volume at runtime
ENV HF_HOME=/data/hf-cache
ENV PYTHONUNBUFFERED=1

# Add venv to PATH so `uv run` and direct invocations work
ENV PATH="/app/.venv/bin:$PATH"

# Healthcheck polls the /health endpoint every 30 s, allows 60 s startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "pgkg.api:app", "--host", "0.0.0.0", "--port", "8000"]
