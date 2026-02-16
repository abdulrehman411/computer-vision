#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
cd "$SCRIPT_DIR"

# Get the venv Python
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python venv not found at $PYTHON_BIN"
    exit 1
fi

exec "$PYTHON_BIN" "$@"
