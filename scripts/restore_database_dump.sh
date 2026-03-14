#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dump-path> <target-database-url>"
  exit 1
fi

DUMP_PATH="$1"
TARGET_DATABASE_URL="$2"

if [ ! -f "$DUMP_PATH" ]; then
  echo "Dump file not found: $DUMP_PATH"
  exit 1
fi

pg_restore \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  --dbname="$TARGET_DATABASE_URL" \
  "$DUMP_PATH"

echo "Database restore complete."
