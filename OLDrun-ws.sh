#!/usr/bin/env bash

Run the EEBO pipeline webservice - though surely it has gone

PYTHONPATH=python/src uvicorn ws.service:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload-dir python/src \
  "$@"

