#!/bin/zsh
cd "/Users/benmills/Stockport_Model" || exit 1
if [ ! -x "./.venv/bin/python" ]; then
  echo "Virtual environment not found at ./.venv/bin/python"
  exit 1
fi
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Run: brew install cloudflared"
  exit 1
fi

PORT="8788"
USERNAME="stockport"
PASSWORD="$("./.venv/bin/python" - <<'PY'
import secrets
print(secrets.token_urlsafe(12))
PY
)"

export STOCKPORT_VIEWER_READ_ONLY=true
export STOCKPORT_VIEWER_BASIC_AUTH_USER="$USERNAME"
export STOCKPORT_VIEWER_BASIC_AUTH_PASSWORD="$PASSWORD"

if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  "./.venv/bin/python" -m viewer.app --host 127.0.0.1 --port "$PORT" >/tmp/stockport_viewer_public.log 2>&1 &
  sleep 2
fi

echo "Public read-only viewer starting..."
echo ""
echo "Username: $USERNAME"
echo "Password: $PASSWORD"
echo ""
echo "Share the public URL shown below with those credentials."
echo "Keep this window open while people are using the app."
echo ""

exec cloudflared tunnel --url "http://127.0.0.1:$PORT"
