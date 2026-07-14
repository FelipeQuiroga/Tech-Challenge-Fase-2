# syntax=docker/dockerfile:1

# ---- builder: instala dependencias com Poetry em um virtualenv isolado ----
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

# Camada de dependencias isolada do codigo-fonte: so invalida com mudancas
# em pyproject.toml/poetry.lock, nao a cada alteracao de codigo.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only main

# ---- runtime: apenas o virtualenv pronto + codigo, sem Poetry/build tools ----
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv ./.venv
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY params.yaml dvc.yaml ./

RUN mkdir -p data/raw data/interim data/processed models \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m"]
CMD ["scripts.train"]