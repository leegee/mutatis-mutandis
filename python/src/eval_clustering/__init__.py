"""
Evaluation clustering framework (parallel system)

This package implements two independent clustering interpretations:

    1. Density clustering (HDBSCAN over embeddings)
    2. Graph clustering (FAISS kNN → community detection)

It is strictly read-only over the existing substrate:

    - events table
    - neighbours table
    - FAISS index
    - Zarr lookup

No outputs are written back to the corpus schema.
"""

from .substrate import Substrate
from .slicing import build_scope

from .hdbscan_runner import run_density_pipeline
from .graph_runner import run_graph_pipeline

from .compare import compare_runs
