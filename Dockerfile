# Anchor Dockerfile — multi-stage, non-root, healthcheck.
# Build:  docker build -t anchor:local .
# Run:    docker run --rm -p 8088:8088 -e MINIMAX_API_KEY_1=... anchor:local

FROM python:3.12-slim AS builder
WORKDIR /build

# System deps for httpx + sqlite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir=/wheels .

FROM python:3.12-slim
WORKDIR /app

# Non-root user for runtime.
RUN groupadd -r anchor && useradd -r -g anchor -d /app anchor

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels anchor \
    && rm -rf /wheels

# Pre-create writable dirs Anchor needs at runtime.
RUN mkdir -p /app/data /app/data/anchor_sessions /home/anchor/logs \
    && chown -R anchor:anchor /app /home/anchor

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    ANCHOR_PORT=8088

USER anchor

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://127.0.0.1:8088/healthz', timeout=3); r.raise_for_status()" || exit 1

CMD ["python", "-m", "uvicorn", "anchor.server:app", \
     "--host", "0.0.0.0", "--port", "8088", "--log-level", "info"]
