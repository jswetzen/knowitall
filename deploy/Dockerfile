# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY server ./server
COPY schema ./schema
COPY client ./client

ENV KNOWITALL_DATA_DIR=/data \
    KNOWITALL_HOST=0.0.0.0 \
    KNOWITALL_PORT=8765 \
    PATH=/opt/venv/bin:$PATH

EXPOSE 8765
VOLUME ["/data"]

CMD ["uvicorn", "--factory", "server.app:create_app", "--host", "0.0.0.0", "--port", "8765"]
