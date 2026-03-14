#!/bin/zsh
cd "/Users/benmills/Stockport_Model" || exit 1
if [ ! -x "./.venv/bin/python" ]; then
  echo "Virtual environment not found at ./.venv/bin/python"
  exit 1
fi

HOST_IP="$(ipconfig getifaddr en0 2>/dev/null)"
if [ -z "$HOST_IP" ]; then
  HOST_IP="$(ipconfig getifaddr en1 2>/dev/null)"
fi
if [ -z "$HOST_IP" ]; then
  HOST_IP="localhost"
fi

echo "Stockport viewer will be available in read-only mode at:"
echo "http://$HOST_IP:8787/"
echo ""
echo "This shared view is browse-only while this window stays running."

exec "./.venv/bin/python" -m viewer.app --host 0.0.0.0 --port 8787 --read-only --open-browser
