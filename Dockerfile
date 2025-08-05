FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev


COPY . /app

FROM python:3.13-slim-bookworm

WORKDIR /app

ARG APP_UID=10001
ARG APP_GID=10001

RUN apt-get update && \
    apt-get install -y --no-install-recommends catdoc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${APP_GID} appgroup && \
    useradd -u ${APP_UID} -g ${APP_GID} -s /usr/sbin/nologin -M appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=appuser:appgroup /app/ ./

USER appuser

CMD ["streamlit", "run", "src/main.py"]