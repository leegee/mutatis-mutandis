#!/usr/bin/env bash
set -e

# --- Config ---
PLANTUML_JAR=lib/plantuml.jar
PLANTUML_URL="https://github.com/plantuml/plantuml/releases/download/v1.2026.2/plantuml.jar"
PUML_DIR=src/diagrams
OUTPUT_DIR=output

echo "===== RUNNING PlantUML ====="
echo "Project root: $(pwd)"

# --- Ensure PlantUML jar exists ---
if [ ! -f "$PLANTUML_JAR" ]; then
    echo "[INFO] PlantUML jar not found at $PLANTUML_JAR"
    echo "[INFO] Downloading from $PLANTUML_URL ..."
    mkdir -p "$(dirname "$PLANTUML_JAR")"
    curl -L -o "$PLANTUML_JAR" "$PLANTUML_URL"
    echo "[INFO] Download complete."
fi
echo "[INFO] Using PlantUML jar: $PLANTUML_JAR"

# --- Compute absolute output directory ---
ABS_OUTPUT=$(realpath "$OUTPUT_DIR")
mkdir -p "$ABS_OUTPUT"
echo "[INFO] Output directory: $ABS_OUTPUT"

# --- Find all PUML files ---
PUML_FILES=$(find "$PUML_DIR" -maxdepth 1 -type f -name "*.puml")
if [ -z "$PUML_FILES" ]; then
    echo "[WARN] No .puml files found in $PUML_DIR"
    exit 0
fi

# --- Process each PUML file ---
for PUML in $PUML_FILES; do
    ABS_PUML=$(realpath "$PUML")
    echo "[INFO] Processing PUML file: $ABS_PUML"

    java -jar "$PLANTUML_JAR" -tsvg -o "$ABS_OUTPUT" "$ABS_PUML"

    SVG_FILE="$ABS_OUTPUT/$(basename "$PUML" .puml).svg"
    echo "[INFO] Generated SVG: $SVG_FILE"
done

echo "[INFO] PlantUML completed for all files."
echo "=============================="
