# ══════════════════════════════════════════════════════════════════════════
# Multimodal Support Chatbot — Production Dockerfile
# Multi-stage build for minimal image size
# ══════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# RapidOCR downloads ~40MB of PyTorch OCR weights into its own models/
# directory on first use. Grant write access to ONLY that directory.
RUN chown -R 1000:1000 /usr/local/lib/python3.11/site-packages/rapidocr/models/

# ── Create non-root user ────────────────────────────────────────────────
# Pre-create the HuggingFace cache directory so the Docker named volume
# inherits appuser ownership on first mount (Docker named volume behaviour).
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

USER appuser

# ── Environment ─────────────────────────────────────────────────────────
# HF_HOME tells all HuggingFace libraries (SentenceTransformers, Docling,
# open_clip) to use the persistent Docker volume for model caching.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

