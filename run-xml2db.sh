#!/usr/bin/env bash

echo "Running the pipeline from TEI XML files..."

cd python
python src/pipeline.py --phase ingest
