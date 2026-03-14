#!/usr/bin/env bash
set -euo pipefail

export STOCKPORT_VIEWER_READ_ONLY="${STOCKPORT_VIEWER_READ_ONLY:-true}"

alembic upgrade head

exec gunicorn \
  viewer.app:application \
  --bind "0.0.0.0:${PORT:-10000}" \
  --workers "${WEB_CONCURRENCY:-2}" \
  --threads "${GUNICORN_THREADS:-4}" \
  --timeout "${GUNICORN_TIMEOUT:-120}"
