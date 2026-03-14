#!/bin/zsh
cd "/Users/benmills/Stockport_Model" || exit 1
if [ ! -x "./.venv/bin/python" ]; then
  echo "Virtual environment not found at ./.venv/bin/python"
  exit 1
fi
exec "./.venv/bin/python" -m viewer.app --read-only --open-browser
