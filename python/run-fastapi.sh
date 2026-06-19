#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$SCRIPT_DIR/python"

cd "$PYTHON_ROOT"

export PYTHONPATH="$PYTHON_ROOT"

# exec uvicorn src.fast_api.app:app \
#     --host 0.0.0.0 \
#     --port 8000

exec uvicorn src.fast_api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload