#!/usr/bin/env bash

bunx concurrently -k -n api,web -c blue,green \
  "cd python/src/passage_search_pipeline && uvicorn passage_search_pipeline.api:app --app-dir src --port 8000" \
  "cd gui/passage-pipeline && bun run dev"
