#!/usr/bin/env bash
set -euo pipefail

FAST_API_PORT=8000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="$SCRIPT_DIR"

cd "$PYTHON_ROOT"
echo [run-fastapi.sh] Enter `pwd`

export PYTHONPATH="$PYTHON_ROOT"

echo [run-fastapi] Calling uvicorn src.fast_api.app:app listening on 0.0.0.0:${FAST_API_PORT}
echo [run-fastapi] See http://127.0.0.1:${FAST_API_PORT}/docs

exec uvicorn src.fast_api.app:app \
    --host 0.0.0.0 \
    --port $FAST_API_PORT \
    --reload