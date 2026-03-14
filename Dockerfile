FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY config ./config
COPY db ./db
COPY features ./features
COPY gates ./gates
COPY governance ./governance
COPY ingestion ./ingestion
COPY models ./models
COPY outputs ./outputs
COPY scoring ./scoring
COPY templates ./templates
COPY training ./training
COPY viewer ./viewer
COPY scripts ./scripts

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

ENV STOCKPORT_VIEWER_READ_ONLY=true \
    PORT=10000

EXPOSE 10000

CMD ["./scripts/render-start.sh"]
