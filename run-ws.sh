#!/usr/bin/env bash

# Run the EEBO pipeline webservice

PYTHONPATH=python/src uvicorn ws.service:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload-dir python/src \
  "$@"

