import sqlite3
import faiss

from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.embedding_cache import EmbeddingCache
from lib.eebo_config import (
    ZARR_PATH,
    FAISS_TIER1_INDEX,
    CORPUS_TIER2_DB_PATH,
)
from tier2_0_concept_events import ZarrEventLookup

from eval_clustering.substrate import Substrate
from eval_clustering.slicing import build_scope
from eval_clustering.hdbscan_runner import run_density_pipeline
from eval_clustering.graph_runner import run_graph_pipeline
from eval_clustering.compare import compare_runs
from eval_clustering.writer import ClusterWriter


def run_all():
    db = sqlite3.connect(CORPUS_TIER2_DB_PATH)

    lookup = ZarrEventLookup(ZARR_PATH)
    index = EeboFaissIndex.load(FAISS_TIER1_INDEX)

    substrate = Substrate(db, lookup, index)
    writer = ClusterWriter(CORPUS_TIER2_DB_PATH)

    concepts = [
        row[0]
        for row in db.execute("""
            SELECT DISTINCT concept
            FROM concepts
            ORDER BY concept
        """)
    ]

    results = {}

    for concept in concepts:

        logger.info(f"Running concept={concept}")

        event_ids = build_scope(
            substrate,
            scope_type="concept",
            scope_value=concept,
        )

        if not event_ids:
            continue

        density = run_density_pipeline(substrate, event_ids)
        graph = run_graph_pipeline(substrate, event_ids)

        metrics = compare_runs(density, graph)

        run_id = writer.write_run(
            concept=concept,
            method="density+graph",
            scope_type="concept",
            scope_value=concept,
            params={
                "n_events": len(event_ids),
            },
        )

        writer.write_memberships(
            run_id=run_id,
            event_ids=density.event_ids,
            cluster_ids=density.labels,
            source="density",
        )

        writer.write_memberships(
            run_id=run_id,
            event_ids=graph.event_ids,
            cluster_ids=graph.labels,
            source="graph",
        )

        writer.write_run_metrics(run_id, metrics)

        results[concept] = {
            "density": density,
            "graph": graph,
            "metrics": metrics,
            "run_id": run_id,
        }

        logger.info(
            f"{concept}: "
            f"{len(event_ids)} events | "
            f"HDBSCAN={len(set(density.labels))} clusters | "
            f"Graph={len(set(graph.labels))} clusters"
        )

    writer.close()

    return results


if __name__ == "__main__":
    run_all()
