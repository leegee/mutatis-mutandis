#!/usr/bin/env bash

bunx concurrently -k -n api,web -c blue,green \
  "uv run --directory python uvicorn passage_search_pipeline.api:app --app-dir src --port 8000 --reload --reload-dir src" \
  "cd frontend && bun run dev"
