#!/usr/bin/env bash
set -euo pipefail

./check.sh

echo "→ Running pipeline"
python src/eebo_parse_tei.py
