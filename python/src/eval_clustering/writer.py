import sqlite3
import json
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


class ClusterWriter:
    """
    Append-only experimental logging layer.

    This is NOT part of the semantic substrate.

    Design intent:
        - runs are immutable
        - every clustering output is reproducible via run_id
        - schema is inspection-first, not storage-minimal
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    # schema: todo move

    def _init_schema(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_run (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT NOT NULL,
            method TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            scope_value TEXT,
            created_at TEXT NOT NULL,
            params_json TEXT NOT NULL
        )
        """)

        cur.execute("""DROP TABLE cluster_membership""");
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_membership (
            run_id TEXT NOT NULL,
            event_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            pub_year INTEGER,
            score REAL,
            PRIMARY KEY (run_id, event_id, source)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_summary (
            run_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            size INTEGER NOT NULL,
            meta_json TEXT,
            PRIMARY KEY (run_id, cluster_id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_edge (
            run_id INTEGER NOT NULL,
            source_event_id INTEGER NOT NULL,
            target_event_id INTEGER NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (run_id, source_event_id, target_event_id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_run_metric (
            run_id INTEGER PRIMARY KEY,
            metrics_json TEXT NOT NULL
        )
        """)

        self.conn.commit()


    def write_run(
        self,
        concept: str,
        method: str,
        scope_type: str,
        scope_value: Optional[str],
        params: Dict[str, Any],
    ) -> int:
        """
        Creates a run and returns run_id.
        """
        cur = self.conn.cursor()

        cur.execute("""
            INSERT INTO cluster_run (
                concept, method, scope_type, scope_value,
                created_at, params_json
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            concept,
            method,
            scope_type,
            scope_value,
            datetime.utcnow().isoformat(),
            json.dumps(params, ensure_ascii=False),
        ))

        self.conn.commit()
        return cur.lastrowid


    def write_memberships(
        self,
        run_id: str,
        event_ids,
        cluster_ids,
        source: str,
        pub_years=None,
        scores=None,
    ):
        """
        One row per event per run.
        """

        event_ids = list(event_ids)
        cluster_ids = list(cluster_ids)

        if pub_years is None:
            pub_years = [None] * len(event_ids)

        if scores is None:
            scores = [None] * len(event_ids)

        rows = [
            (run_id, int(eid), int(cid), source, py, sc)
            for eid, cid, py, sc in zip(event_ids, cluster_ids, pub_years, scores)
        ]

        self.conn.executemany("""
            INSERT INTO cluster_membership (
                run_id, event_id, cluster_id, source, pub_year, score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, rows)

        self.conn.commit()


    def write_graph_edges(
        self,
        run_id: int,
        graph: Dict[int, list],
    ):
        rows = []

        for src, edges in graph.items():
            for tgt, w in edges:
                rows.append((run_id, int(src), int(tgt), float(w)))

        self.conn.executemany("""
            INSERT OR IGNORE INTO cluster_edge (
                run_id, source_event_id, target_event_id, weight
            ) VALUES (?, ?, ?, ?)
        """, rows)

        self.conn.commit()


    def write_cluster_summaries(
        self,
        run_id: int,
        cluster_sizes: Dict[int, int],
        meta: Optional[Dict[int, Dict]] = None,
    ):
        rows = [
            (
                run_id,
                int(cid),
                int(size),
                json.dumps(meta.get(cid) if meta else None),
            )
            for cid, size in cluster_sizes.items()
        ]

        self.conn.executemany("""
            INSERT INTO cluster_summary (
                run_id, cluster_id, size, meta_json
            ) VALUES (?, ?, ?, ?)
        """, rows)

        self.conn.commit()


    def write_run_metrics(self, run_id: int, metrics: Dict[str, Any]):
        self.conn.execute("""
            INSERT OR REPLACE INTO cluster_run_metric (
                run_id, metrics_json
            ) VALUES (?, ?)
        """, (
            run_id,
            json.dumps(metrics, ensure_ascii=False),
        ))

        self.conn.commit()

    def close(self):
        self.conn.close()
