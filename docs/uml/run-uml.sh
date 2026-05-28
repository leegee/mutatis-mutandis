#!/usr/bin/env bash
set -e

PLANTUML_JAR=lib/plantuml.jar
PUML_DIR=src/diagrams
OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

echo "===== RUNNING PlantUML ====="

cd "$PUML_DIR"

for PUML in *.puml; do
    echo "[INFO] Processing $PUML"

    java -jar ../../"$PLANTUML_JAR" \
        -tsvg \
        -o ../../"$OUTPUT_DIR" \
        "$PUML"
done

echo "[INFO] Done"