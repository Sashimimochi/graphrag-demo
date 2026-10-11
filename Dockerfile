# Resolve and install dependencies from the committed uv lockfile.
FROM ghcr.io/astral-sh/uv:0.12.19-python3.12-trixie-slim AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# --- Runtime stage ---
FROM python:3.12-slim-trixie

WORKDIR /app

# Native libraries required by image/document processing dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Keep the environment outside /app because docker-compose bind-mounts the source tree there.
COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
