# tier1/tier1_corpus2events.py

from __future__ import annotations

import argparse
import math
import os
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import torch

from embedding.macberth_worker import (
    DocBuffer,
    EmbeddedVector,
    MacBERThEventEmbedder,
    TokenRow,
)
from lib.corpus_config import (
    CONCEPT_SETS,
    EMBED_BATCH_SIZE,
    LANCE_INDEXES_DIR,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import load_macberth
from tier1.db_observation_backend import (
    allocate_event_ids,
    create_events_table,
    insert_events,
    migrate_events_table,
)


os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)
os.environ.setdefault(
    "OMP_NUM_THREADS",
    "4",
)
os.environ.setdefault(
    "MKL_NUM_THREADS",
    "4",
)
os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "2",
)


ACTIVE_SCALES = ("local",)

LANCE_MODEL_NAME = "macberth"
LANCE_BUCKET_SIZE = 50

VECTOR_INDEX_TYPE = "IVF_FLAT"
VECTOR_INDEX_METRIC = "cosine"

MIN_VECTOR_INDEX_PARTITIONS = 1
MAX_VECTOR_INDEX_PARTITIONS = 256


def normalise_token(token: str) -> str:
    return unicodedata.normalize(
        "NFKC",
        token,
    ).strip().lower()


def seed_forms() -> set[str]:
    forms: set[str] = set()

    for rule in CONCEPT_SETS.values():
        forms.update(
            normalise_token(form)
            for form in rule["forms"]
        )

    return forms


def false_positive_forms() -> set[str]:
    forms: set[str] = set()

    for rule in CONCEPT_SETS.values():
        forms.update(
            normalise_token(form)
            for form in rule["false_positives"]
        )

    return forms


SEED_FORMS = seed_forms()
FALSE_POSITIVE_FORMS = false_positive_forms()


def is_seed(token: str) -> bool:
    value = normalise_token(token)

    return (
        value in SEED_FORMS
        and value not in FALSE_POSITIVE_FORMS
    )


def year_bucket(
    year: int,
) -> tuple[int, int]:
    start = (
        year // LANCE_BUCKET_SIZE
    ) * LANCE_BUCKET_SIZE

    return (
        start,
        start + LANCE_BUCKET_SIZE - 1,
    )


def lance_table_name(
    scale: str,
    year: int,
) -> str:
    start, end = year_bucket(year)

    return (
        f"{scale}__{LANCE_MODEL_NAME}__"
        f"{start:04d}_{end:04d}"
    )


def vector_index_partitions(
    row_count: int,
) -> int:
    """
    Size num_partitions from the table's actual row count.
    """
    if row_count <= 0:
        raise ValueError(
            "row_count must be positive"
        )

    return max(
        MIN_VECTOR_INDEX_PARTITIONS,
        min(
            MAX_VECTOR_INDEX_PARTITIONS,
            int(math.sqrt(row_count)),
        ),
    )


@dataclass(slots=True)
class Observation:
    event_id: int | None
    corpus: str
    doc_id: str
    token: str
    token_idx: int
    pub_year: int | None
    scale: str
    window_id: int
    window_token_pos: int


@dataclass(slots=True)
class EmbeddedObservation:
    observation: Observation
    vectors: dict[str, np.ndarray]


class EventWriter:
    def __init__(
        self,
        conn,
        lance_root: Path,
    ) -> None:
        self.conn = conn
        self.lance_root = lance_root
        self.lance = lancedb.connect(
            str(lance_root)
        )
        self.tables: dict[str, object] = {}


    def repair_lance(
        self,
        observations: list[EmbeddedObservation],
    ) -> int:
        if not observations:
            return 0

        self._resolve_repair_event_ids(
            observations,
        )

        return self.write_lance(
            observations,
        )

    def index_existing_tables(self) -> None:
        prefixes = tuple(
            f"{scale}__{LANCE_MODEL_NAME}__"
            for scale in ACTIVE_SCALES
        )

        for table_name in sorted(
            name
            for name in self.lance.list_tables().tables
            if name.startswith(prefixes)
        ):
            self.tables[table_name] = (
                self.lance.open_table(table_name)
            )

        self.build_indexes()

    def build_indexes(self) -> None:
        """
        Rebuild indexes for tables that are missing an index
        or contain unindexed rows.
        """
        expected_indexes = {
            "vector_idx",
            "event_id_idx",
            "year_idx",
            "embedding_model_idx",
        }

        for table_name, table in self.tables.items():
            row_count = table.count_rows()

            if row_count == 0:
                continue

            indices = {
                index.name: index
                for index in table.list_indices()
            }

            if (
                expected_indexes <= indices.keys()
                and all(
                    indices[name].num_unindexed_rows == 0
                    for name in expected_indexes
                )
            ):
                logger.info(
                    "[tier1] indexes already complete "
                    "for %s (%d rows)",
                    table_name,
                    row_count,
                )
                continue

            num_partitions = (
                vector_index_partitions(
                    row_count
                )
            )

            logger.info(
                "[tier1] rebuilding indexes for %s "
                "(%d rows, %d partitions)",
                table_name,
                row_count,
                num_partitions,
            )

            table.create_index(
                metric=VECTOR_INDEX_METRIC,
                index_type=VECTOR_INDEX_TYPE,
                vector_column_name="vector",
                num_partitions=num_partitions,
                replace=True,
            )

            table.create_scalar_index(
                "event_id",
                index_type="BTREE",
                replace=True,
            )

            table.create_scalar_index(
                "year",
                index_type="BTREE",
                replace=True,
            )

            table.create_scalar_index(
                "embedding_model",
                index_type="BTREE",
                replace=True,
            )


    def _resolve_repair_event_ids(
        self,
        observations: list[EmbeddedObservation],
    ) -> None:
        """
        Resolve repair observations to their existing PostgreSQL event IDs
        using normalized contextual provenance.
        """
        if not observations:
            return

        for embedded in observations:
            observation = embedded.observation

            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id
                    FROM events
                    WHERE corpus = %s
                    AND doc_id = %s
                    AND token_idx = %s
                    AND scale = %s
                    AND window_id = %s
                    AND window_token_pos = %s
                    """,
                    (
                        observation.corpus,
                        observation.doc_id,
                        observation.token_idx,
                        observation.scale,
                        observation.window_id,
                        observation.window_token_pos,
                    ),
                )

                rows = cur.fetchall()

            if not rows:
                raise RuntimeError(
                    "Lance repair could not find an existing "
                    "PostgreSQL event for observation "
                    f"{observation.corpus}/"
                    f"{observation.doc_id}/"
                    f"{observation.token_idx}/"
                    f"{observation.scale}/"
                    f"{observation.window_id}/"
                    f"{observation.window_token_pos}"
                )

            if len(rows) > 1:
                raise RuntimeError(
                    "Lance repair found multiple PostgreSQL events "
                    "for the same observation provenance: "
                    f"{observation.corpus}/"
                    f"{observation.doc_id}/"
                    f"{observation.token_idx}/"
                    f"{observation.scale}/"
                    f"{observation.window_id}/"
                    f"{observation.window_token_pos}"
                )

            observation.event_id = int(rows[0][0])

    def write_lance(
        self,
        observations: list[EmbeddedObservation],
    ) -> int:
        rows_by_table: dict[
            str,
            list[dict],
        ] = defaultdict(list)

        for embedded in observations:
            observation = embedded.observation

            if observation.event_id is None:
                raise RuntimeError(
                    "Cannot write an observation to Lance "
                    "without an authoritative PostgreSQL event ID."
                )

            if observation.pub_year is None:
                raise ValueError(
                    f"Observation {observation.event_id} "
                    "has no publication year and cannot be "
                    "assigned to a chronological Lance table."
                )

            for scale, vector in embedded.vectors.items():
                if scale not in {"local", "medium", "broad"}:
                    raise RuntimeError(
                        f"Unknown embedding scale {scale!r}: "
                        f"event_id={observation.event_id}"
                    )

                table_name = lance_table_name(
                    scale,
                    observation.pub_year,
                )

                rows_by_table[table_name].append(
                    {
                        "event_id": ( observation.event_id ),
                        "year": observation.pub_year,
                        "embedding_model": ( LANCE_MODEL_NAME ),
                        "vector": ( embedded.vectors[scale] .tolist() ),
                    }
                )

        written_event_ids: set[int] = set()

        for (
            table_name,
            rows,
        ) in rows_by_table.items():
            table = self._open_table(
                table_name,
                vector_dimensions=len(
                    rows[0]["vector"]
                ),
            )

            existing_ids = (
                self._existing_lance_ids(
                    table,
                    {
                        row["event_id"]
                        for row in rows
                    },
                )
            )

            new_rows = [
                row
                for row in rows
                if row["event_id"]
                not in existing_ids
            ]

            if new_rows:
                table.add(
                    new_rows,
                    mode="append",
                )

                written_event_ids.update(
                    row["event_id"]
                    for row in new_rows
                )

        return len(written_event_ids)

    def _existing_lance_ids(
        self,
        table,
        event_ids: set[int],
    ) -> set[int]:
        if not event_ids:
            return set()

        arrow = table.to_arrow()

        existing = set(
            arrow.column(
                "event_id"
            ).to_pylist()
        )

        return existing.intersection(
            event_ids
        )

    def _open_table(
        self,
        table_name: str,
        *,
        vector_dimensions: int,
    ):
        if table_name in self.tables:
            return self.tables[table_name]

        table_names = set(
            self.lance.list_tables().tables
        )

        if table_name in table_names:
            table = self.lance.open_table(
                table_name
            )

            self.tables[table_name] = table

            return table

        logger.info(
            "[tier1] creating Lance table: %s",
            table_name,
        )

        table = self.lance.create_table(
            table_name,
            schema=pa.schema(
                [
                    pa.field(
                        "event_id",
                        pa.uint64(),
                    ),
                    pa.field(
                        "year",
                        pa.int32(),
                    ),
                    pa.field(
                        "embedding_model",
                        pa.string(),
                    ),
                    pa.field(
                        "vector",
                        pa.list_(
                            pa.float32(),
                            vector_dimensions,
                        ),
                    ),
                ]
            ),
        )

        self.tables[table_name] = table

        return table


class CorpusProcessor:
    def __init__(
        self,
        conn,
        embedder: MacBERThEventEmbedder,
        writer: EventWriter,
        *,
        neighbour_radius: int = 256,
        report_every: int = 25,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.writer = writer
        self.neighbour_radius = neighbour_radius
        self.report_every = report_every

    def process(
        self,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
        embed_all: bool = False,
    ) -> None:
        if embed_all:
            documents = self._find_all_documents(
                corpus=corpus,
                doc_id=doc_id,
            )
        else:
            documents = self._find_seed_documents(
                corpus=corpus,
                doc_id=doc_id,
            )

        logger.info(
            "[tier1] selected documents: %d",
            len(documents),
        )

        if documents:
            logger.info(
                "[tier1] first selected documents: %s",
                documents[:5],
            )

        for number, (
            document_corpus,
            document_id,
        ) in enumerate(
            documents,
            start=1,
        ):
            started = time.perf_counter()

            document = self._load_document(
                document_corpus,
                document_id,
            )

            if document is None:
                continue

            if embed_all:
                seed_count = sum(
                    1
                    for row in document.rows
                    if is_seed(row.token)
                )

                target_positions = set(
                    range(len(document.rows))
                )
            else:
                seed_positions = {
                    position
                    for position, row
                    in enumerate(document.rows)
                    if is_seed(row.token)
                }

                if not seed_positions:
                    continue

                seed_count = len(seed_positions)

                target_positions = (
                    self._select_neighbours(
                        seed_positions,
                        len(document.rows),
                    )
                )

            embedded_by_position = (
                self.embedder.embed_document_targets(
                    document=document,
                    target_positions=target_positions,
                )
            )

            observations = self._build_observations(
                document=document,
                embedded_by_position=embedded_by_position,
            )

            event_ids = self._create_events(
                observations,
            )

            if len(event_ids) != len(observations):
                raise RuntimeError(
                    "Number of allocated event IDs does not "
                    "match number of observations."
                )

            embedded_observations = (
                self._attach_vectors(
                    observations=observations,
                    embedded_by_position=embedded_by_position,
                    document=document
                )
            )

            written = self.writer.write_lance(
                embedded_observations,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            logger.info(
                "[tier1] %3d/%-3d %-4s %-15s "
                "seeds=%3d observations=%5d "
                "written=%5d elapsed=%7.2fs",
                number,
                len(documents),
                document.corpus,
                document.doc_id,
                seed_count,
                len(observations),
                written,
                elapsed,
            )

            if (
                number
                % self.report_every
                == 0
            ):
                logger.info(
                    "[tier1] processed %d documents",
                    number,
                )

    def repair(
        self,
        *,
        corpus: str,
        doc_id: str,
    ) -> None:
        """
        Regenerate Lance vectors for one existing document without
        creating new PostgreSQL events.
        """
        started = time.perf_counter()

        logger.info(
            "[tier1] repair: %s/%s",
            corpus,
            doc_id,
        )

        document = self._load_document(
            corpus,
            doc_id,
        )

        if document is None:
            raise RuntimeError(
                f"Document not found: "
                f"{corpus}/{doc_id}"
            )

        seed_positions = {
            position
            for position, row
            in enumerate(document.rows)
            if is_seed(row.token)
        }

        if not seed_positions:
            raise RuntimeError(
                f"No seed occurrences found: "
                f"{corpus}/{doc_id}"
            )

        target_positions = (
            self._select_neighbours(
                seed_positions,
                len(document.rows),
            )
        )

        logger.info(
            "[tier1] repair: seeds=%d observations=%d",
            len(seed_positions),
            len(target_positions),
        )

        embedded_by_position = (
            self.embedder.embed_document_targets(
                document=document,
                target_positions=target_positions,
            )
        )

        embedded_observations = (
            self._build_embedded_observations(
                document=document,
                embedded_by_position=embedded_by_position,
            )
        )

        written = self.writer.repair_lance(
            embedded_observations,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.info(
            "[tier1] repair complete: %-4s %-15s "
            "seeds=%3d observations=%5d "
            "lance_written=%5d elapsed=%7.2fs",
            corpus,
            doc_id,
            len(seed_positions),
            len(embedded_observations),
            written,
            elapsed,
        )

    def _load_document(
        self,
        corpus: str,
        doc_id: str,
    ) -> DocBuffer | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.corpus,
                    t.doc_id,
                    t.token_idx,
                    t.token,
                    d.pub_year
                FROM tokens AS t
                JOIN documents AS d
                  ON d.corpus = t.corpus
                 AND d.doc_id = t.doc_id
                WHERE t.corpus = %s
                  AND t.doc_id = %s
                ORDER BY t.token_idx
                """,
                (
                    corpus,
                    doc_id,
                ),
            )

            rows = cur.fetchall()

        if not rows:
            return None

        token_indices = [
            row[2]
            for row in rows
        ]

        expected_indices = list(
            range(len(rows))
        )

        if token_indices != expected_indices:
            raise RuntimeError(
                f"Document {corpus}/{doc_id} has non-contiguous "
                "token_idx values; cannot safely map embedding "
                "positions to corpus token indices."
            )

        return DocBuffer(
            corpus=corpus,
            doc_id=doc_id,
            pub_year=rows[0][4],
            rows=[
                TokenRow(
                    corpus=row[0],
                    doc_id=row[1],
                    token_idx=row[2],
                    token=row[3],
                    pub_year=row[4],
                )
                for row in rows
            ],
        )


    def _build_observations(
        self,
        *,
        document: DocBuffer,
        embedded_by_position: dict[int, EmbeddedVector],
    ) -> list[Observation]:
        """
        Construct complete observations from the actual MacBERTh
        embedding results.

        Window provenance is established before PostgreSQL event IDs
        are allocated.
        """
        scale = self.embedder.scale

        if scale not in {"local", "medium", "broad"}:
            raise RuntimeError(
                f"Unsupported embedding scale: {scale!r}"
            )

        observations: list[Observation] = []

        for position, embedded in sorted(
            embedded_by_position.items()
        ):
            row = document.rows[position]

            if embedded.window_id is None:
                raise RuntimeError(
                    f"Embedding at position {position} has no window_id"
                )

            if embedded.window_token_pos is None:
                raise RuntimeError(
                    f"Embedding at position {position} has no "
                    "window_token_pos"
                )

            observations.append(
                Observation(
                    event_id=None,
                    corpus=document.corpus,
                    doc_id=document.doc_id,
                    token=row.token,
                    token_idx=row.token_idx,
                    pub_year=row.pub_year,
                    scale=scale,
                    window_id=int(embedded.window_id),
                    window_token_pos=int(
                        embedded.window_token_pos
                    ),
                )
            )

        return observations


    def _create_events(
        self,
        observations: list[Observation],
    ) -> list[int]:
        """
        Resolve existing events or create new authoritative event IDs.

        Observation provenance is the natural event identity:

            corpus + doc_id + token_idx + scale +
            window_id + window_token_pos

        Existing observations reuse their existing event IDs.
        Only genuinely new observations consume event IDs.
        """
        if not observations:
            return []

        scale = self.embedder.scale

        if scale not in {"local", "medium", "broad"}:
            raise RuntimeError(
                f"Unsupported embedding scale: {scale!r}"
            )

        for observation in observations:
            if observation.event_id is not None:
                raise RuntimeError(
                    "Cannot create or resolve an observation that already "
                    f"has event_id={observation.event_id}"
                )

            if observation.pub_year is None:
                raise ValueError(
                    "Cannot create an event without publication year: "
                    f"{observation.corpus}/"
                    f"{observation.doc_id}/"
                    f"{observation.token_idx}"
                )

            if observation.scale != scale:
                raise RuntimeError(
                    "Observation scale does not match embedder scale: "
                    f"{observation.scale!r} != {scale!r}"
                )

        # Resolve existing observations in one query.
        existing: dict[
            tuple[str, str, int, str, int, int],
            int,
        ] = {}

        with self.conn.cursor() as cur:
            for observation in observations:
                cur.execute(
                    """
                    SELECT event_id
                    FROM events
                    WHERE corpus = %s
                    AND doc_id = %s
                    AND token_idx = %s
                    AND scale = %s
                    AND window_id = %s
                    AND window_token_pos = %s
                    """,
                    (
                        observation.corpus,
                        observation.doc_id,
                        observation.token_idx,
                        observation.scale,
                        observation.window_id,
                        observation.window_token_pos,
                    ),
                )

                rows = cur.fetchall()

                if len(rows) > 1:
                    raise RuntimeError(
                        "Multiple PostgreSQL events exist for the same "
                        "observation provenance: "
                        f"{observation.corpus}/"
                        f"{observation.doc_id}/"
                        f"{observation.token_idx}/"
                        f"{observation.scale}/"
                        f"{observation.window_id}/"
                        f"{observation.window_token_pos}"
                    )

                if rows:
                    existing[
                        (
                            observation.corpus,
                            observation.doc_id,
                            observation.token_idx,
                            observation.scale,
                            observation.window_id,
                            observation.window_token_pos,
                        )
                    ] = int(rows[0][0])

        new_observations: list[Observation] = []

        for observation in observations:
            key = (
                observation.corpus,
                observation.doc_id,
                observation.token_idx,
                observation.scale,
                observation.window_id,
                observation.window_token_pos,
            )

            existing_event_id = existing.get(key)

            if existing_event_id is not None:
                observation.event_id = existing_event_id
            else:
                new_observations.append(observation)

        # Allocate IDs only for genuinely new observations.
        if new_observations:
            event_ids = allocate_event_ids(
                self.conn,
                len(new_observations),
            )

            if len(event_ids) != len(new_observations):
                raise RuntimeError(
                    "PostgreSQL allocated an unexpected number of event IDs: "
                    f"expected {len(new_observations)}, "
                    f"got {len(event_ids)}"
                )

            for observation, event_id in zip(
                new_observations,
                event_ids,
                strict=True,
            ):
                observation.event_id = int(event_id)

            insert_events(
                self.conn,
                event_id=[
                    int(observation.event_id)
                    for observation in new_observations
                ],
                corpus=[
                    observation.corpus
                    for observation in new_observations
                ],
                doc_id=[
                    observation.doc_id
                    for observation in new_observations
                ],
                token=[
                    observation.token
                    for observation in new_observations
                ],
                token_idx=[
                    observation.token_idx
                    for observation in new_observations
                ],
                pub_year=[
                    observation.pub_year
                    for observation in new_observations
                ],
                scale=[
                    observation.scale
                    for observation in new_observations
                ],
                window_id=[
                    observation.window_id
                    for observation in new_observations
                ],
                window_token_pos=[
                    observation.window_token_pos
                    for observation in new_observations
                ],
            )

            self.conn.commit()

        return [
            int(observation.event_id)
            for observation in observations
        ]

    def _attach_vectors(
        self,
        observations: list[Observation],
        embedded_by_position: dict[int, EmbeddedVector],
        document: DocBuffer,
    ) -> list[EmbeddedObservation]:
        """Attach embedding vectors to observations."""

        token_position = {
            row.token_idx: position
            for position, row in enumerate(document.rows)
        }

        embedded_observations: list[EmbeddedObservation] = []

        for observation in observations:
            position = token_position.get(
                observation.token_idx
            )

            if position is None:
                raise ValueError(
                    f"Observation token_idx "
                    f"{observation.token_idx} "
                    f"not found in document "
                    f"{document.doc_id}"
                )

            embedded = embedded_by_position.get(
                position
            )

            if embedded is None:
                raise ValueError(
                    f"No embedding found for document "
                    f"position {position} "
                    f"(token_idx={observation.token_idx}) "
                    f"in document {document.doc_id}"
                )

            embedded_observations.append(
                EmbeddedObservation(
                    observation=observation,
                    vectors={
                        self.embedder.scale: embedded.vector,
                    },
                )
            )

        return embedded_observations


    def _build_embedded_observations(
        self,
        *,
        document: DocBuffer,
        embedded_by_position: dict[int, EmbeddedVector],
    ) -> list[EmbeddedObservation]:
        observations = self._build_observations(
            document=document,
            embedded_by_position=embedded_by_position,
        )

        return self._attach_vectors(
            observations=observations,
            embedded_by_position=embedded_by_position,
            document=document,
        )


    def _find_all_documents(
        self,
        *,
        corpus: str | None,
        doc_id: str | None,
    ) -> list[tuple[str, str]]:
        clauses: list[str] = []
        params: list[object] = []

        if corpus is not None:
            clauses.append(
                "t.corpus = %s"
            )
            params.append(corpus)

        if doc_id is not None:
            clauses.append(
                "t.doc_id = %s"
            )
            params.append(doc_id)

        where = ""

        if clauses:
            where = (
                "WHERE "
                + " AND ".join(clauses)
            )

        sql = f"""
            SELECT
                t.corpus,
                t.doc_id
            FROM tokens AS t
            {where}
            GROUP BY
                t.corpus,
                t.doc_id
            ORDER BY
                t.corpus,
                t.doc_id
        """

        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                params,
            )

            return cur.fetchall()


    def _find_seed_documents(
        self,
        *,
        corpus: str | None,
        doc_id: str | None,
    ) -> list[tuple[str, str]]:
        clauses = [
            "lower(t.token) = ANY(%s)",
        ]

        params: list[object] = [
            sorted(
                SEED_FORMS
                - FALSE_POSITIVE_FORMS
            ),
        ]

        if corpus is not None:
            clauses.append(
                "t.corpus = %s"
            )
            params.append(corpus)

        if doc_id is not None:
            clauses.append(
                "t.doc_id = %s"
            )
            params.append(doc_id)

        sql = f"""
            SELECT DISTINCT
                t.corpus,
                t.doc_id
            FROM tokens AS t
            WHERE {" AND ".join(clauses)}
            ORDER BY
                t.corpus,
                t.doc_id
        """

        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                params,
            )

            return cur.fetchall()

    def _select_neighbours(
        self,
        seed_positions: set[int],
        token_count: int,
    ) -> set[int]:
        positions: set[int] = set()

        for seed_position in seed_positions:
            start = max(
                0,
                seed_position
                - self.neighbour_radius,
            )

            end = min(
                token_count,
                seed_position
                + self.neighbour_radius
                + 1,
            )

            positions.update(
                range(start, end)
            )

        return positions


def parse_repair_target(
    value: str,
) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError(
            "repair target must be CORPUS/DOC_ID"
        )

    corpus, doc_id = value.split(
        "/",
        1,
    )

    if not corpus or not doc_id:
        raise argparse.ArgumentTypeError(
            "repair target must be CORPUS/DOC_ID"
        )

    return corpus, doc_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Tier 1 observations directly from "
            "PostgreSQL through MacBERTh into Lance."
        )
    )

    parser.add_argument(
        "--corpus",
        default=None,
    )

    parser.add_argument(
        "--doc-id",
        default=None,
    )

    parser.add_argument(
        "--migrate-events",
        action="store_true",
        help=(
            "Migrate the existing events table to the normalized "
            "scale/window provenance schema and exit."
        ),
    )

    parser.add_argument(
        "--embed-all",
        action="store_true",
        help=(
            "Embed every token in each selected document instead "
            "of only neighbourhoods of concept seeds."
        ),
    )

    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Rebuild incomplete indexes on existing "
            "active-scale Lance tables"
        ),
    )

    parser.add_argument(
        "--repair",
        type=parse_repair_target,
        metavar="CORPUS/DOC_ID",
        help=(
            "Regenerate Lance vectors for one document "
            "without modifying PostgreSQL events."
        ),
    )

    parser.add_argument(
        "--neighbour-radius",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        help=(
            "Replace target tokens with [MASK] "
            "before embedding."
        ),
    )

    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help=(
            "Skip the post-run index (re)build step."
        ),
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=Path(
            LANCE_INDEXES_DIR
        ),
    )

    args = parser.parse_args()

    if args.repair is not None and (
        args.corpus is not None
        or args.doc_id is not None
    ):
        parser.error(
            "--repair cannot be combined with "
            "--corpus or --doc-id"
        )

    return args


def main() -> None:
    args = parse_args()

    torch.set_num_threads(
        int(
            os.environ.get(
                "OMP_NUM_THREADS",
                "4",
            )
        )
    )

    torch.set_num_interop_threads(1)

    conn = get_connection()

    if args.migrate_events:
        migrate_events_table(conn)
        conn.close()
        return

    create_events_table(conn)

    if args.index_only:
        writer = EventWriter(
            conn,
            args.lance_root,
        )
        writer.index_existing_tables()
        conn.close()
        return

    try:
        mac = load_macberth()

        embedder = MacBERThEventEmbedder(
            conn,
            mac,
            batch_size=args.batch_size,
            scale="local",
            mask_targets=args.mask,
        )

        writer = EventWriter(
            conn,
            args.lance_root,
        )

        processor = CorpusProcessor(
            conn,
            embedder,
            writer,
            neighbour_radius=args.neighbour_radius,
            report_every=args.report_every,
        )

        if args.repair is not None:
            corpus, doc_id = args.repair

            processor.repair(
                corpus=corpus,
                doc_id=doc_id,
            )
        else:
            processor.process(
                corpus=args.corpus,
                doc_id=args.doc_id,
                embed_all=args.embed_all,
            )

        if args.skip_indexing:
            logger.info(
                "[tier1] --skip-indexing set; leaving "
                "index (re)build for a later run"
            )
        else:
            writer.build_indexes()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
