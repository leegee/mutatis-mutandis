#!/usr/bin/env bash

echo "Running the uvicorn and bun dev servers in parallel"
echo "CTRL+C to quit."

set -m

(
    cd ./python/src/passage_search_pipeline || exit 1
    exec uvicorn passage_search_pipeline.api:app --app-dir src --port 8000
) &
API_PID=$!

(
    cd ./gui/passage-pipeline || exit 1
    exec bun dev
) &
GUI_PID=$!

cleanup() {
    echo
    echo "Stopping servers..."

    trap - INT TERM EXIT

    kill -TERM "$API_PID" "$GUI_PID" 2>/dev/null || true

    # Give servers a moment to shut down cleanly.
    sleep 1

    kill -KILL "$API_PID" "$GUI_PID" 2>/dev/null || true

    wait "$API_PID" "$GUI_PID" 2>/dev/null || true
    echo "Servers stopped."
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

# Wait until either server exits.
wait -n "$API_PID" "$GUI_PID"
STATUS=$?

echo "A server exited (status $STATUS); shutting down the other."
exit "$STATUS"
