#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set -a
if [ -f ".env" ]; then
  . ./.env
fi
set +a

if [ -z "${STOCKPORT_DATABASE_URL:-}" ]; then
  echo "STOCKPORT_DATABASE_URL is not set. Add it to .env or export it in your shell."
  exit 1
fi

mkdir -p artifacts
OUTPUT_PATH="${1:-artifacts/stockport_model_$(date +%Y%m%d_%H%M%S).dump}"

pg_dump \
  "$STOCKPORT_DATABASE_URL" \
  --format=custom \
  --no-owner \
  --no-privileges \
  --file "$OUTPUT_PATH"

echo "Database dump written to $OUTPUT_PATH"
