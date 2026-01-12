#!/usr/bin/env bash
set -euo pipefail

SRC="src"
PYTHON="python"
ENTRY="$SRC/eebo_parse_tei.py"

echo "→ Ruff"
ruff check "$SRC"

echo "→ Mypy"
mypy "$SRC"

echo "→ Pyright"
pyright "$SRC"

echo "✔ All checks passed"

if [[ "${1:-}" == "--smoke" ]]; then
    echo "→ Smoke test"
    "$PYTHON" "$ENTRY" --limit 5
else
    echo "→ Running full ingestion pipeline"
    "$PYTHON" "$ENTRY"
fi

