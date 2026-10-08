from pathlib import Path

from lib.corpus_config import (
    ACTIVE_SCALES,
    LANCE_MODEL_NAME,
    LANCE_BUCKET_SIZE
)
from lib.corpus_logging import logger

from tier1.models import *
from tier1.vector_writer import VectorWriter

class EventWriter:
    """
    Owns PostgreSQL event provenance and delegates vector persistence to
    VectorWriter.

    Token and phrase observations share the event table and Lance event
    space, but retain distinct PostgreSQL provenance through span_end_idx.
    """

    def __init__(
        self,
        conn,
        lance_root: Path,
    ) -> None:
        self.conn = conn
        self.lance_root = lance_root

        self.vector_writers: dict[str, VectorWriter] = {
            scale: VectorWriter(
                lance_root,
                scale=scale,
                model_name=LANCE_MODEL_NAME,
                bucket_size=LANCE_BUCKET_SIZE,
            )
            for scale in ACTIVE_SCALES
        }

    def _writer_for_scale(
        self,
        scale: str,
    ) -> VectorWriter:
        writer = self.vector_writers.get(scale)

        if writer is None:
            writer = VectorWriter(
                self.lance_root,
                scale=scale,
                model_name=LANCE_MODEL_NAME,
                bucket_size=LANCE_BUCKET_SIZE,
            )

            self.vector_writers[scale] = writer

        return writer

    def purge_orphans(
        self,
        conn,
        *,
        year_range=None,
        apply=False,
    ):
        return {
            scale: writer.purge_orphans(
                conn,
                year_range=year_range,
                apply=apply,
            )
            for scale, writer in self.vector_writers.items()
        }

    def write(
        self,
        observations: list[AnyEmbedded],
    ) -> int:
        if not observations:
            return 0

        event_ids = allocate_event_ids(
            self.conn,
            len(observations),
        )

        for embedded, event_id in zip(
            observations,
            event_ids,
        ):
            embedded.observation.event_id = event_id

        self._write_postgres(observations)
        self._write_lance(observations)

        return len(observations)

    def repair_lance(
        self,
        observations: list[AnyEmbedded],
    ) -> int:
        if not observations:
            return 0

        self._resolve_repair_event_ids(observations)

        return self._write_lance(observations)

    def index_existing_tables(self) -> None:
        for writer in self.vector_writers.values():
            writer.index_existing_tables()

    def build_indexes(self) -> None:
        for writer in self.vector_writers.values():
            writer.build_indexes()

    def _write_postgres(
        self,
        observations: list[AnyEmbedded],
    ) -> None:
        if not observations:
            return

        if any(
            embedded.observation.event_id is None
            for embedded in observations
        ):
            raise RuntimeError(
                "Cannot persist observations before PostgreSQL "
                "event IDs have been allocated."
            )

        is_phrase = isinstance(
            observations[0],
            EmbeddedSpanObservation,
        )

        if any(
            isinstance(embedded, EmbeddedSpanObservation) != is_phrase
            for embedded in observations
        ):
            raise RuntimeError( "Cannot write mixed token and phrase observations in a single batch." )


        for item in observations:
            obs = item.observation
            if obs.token_idx is None:
                raise RuntimeError(
                    f"NULL token_idx before PostgreSQL: "
                    f"event_id={obs.event_id!r}, "
                    f"corpus={obs.corpus!r}, "
                    f"doc_id={obs.doc_id!r}, "
                    f"token={obs.token!r}, "
                    f"pub_year={obs.pub_year!r}, "
                    f"observation={obs!r}"
                )

        if is_phrase:
            phrase_observations = [
                embedded
                for embedded in observations
                if isinstance(embedded, EmbeddedSpanObservation)
            ]

            insert_events(
                self.conn,
                event_id=[o.observation.event_id for o in observations],
                corpus=[o.observation.corpus for o in observations],
                doc_id=[o.observation.doc_id for o in observations],
                # FIX: SpanObservation has no `.token`; the span text is
                # `.phrase`. (The original read `.token` and raised
                # AttributeError on the first phrase write.)
                token=[o.observation.phrase for o in observations],
                token_idx=[o.observation.token_idx for o in observations],
                span_end_idx=[o.observation.span_end_idx for o in observations],
                pub_year=[o.observation.pub_year for o in observations],
                local_window_id=[o.observation.local_window_id for o in observations],
                local_window_token_pos=[
                    o.observation.local_window_token_pos for o in observations
                ],
                medium_window_id=[o.observation.medium_window_id for o in observations],
                medium_window_token_pos=[
                    o.observation.medium_window_token_pos for o in observations
                ],
                broad_window_id=[o.observation.broad_window_id for o in observations],
                broad_window_token_pos=[
                    o.observation.broad_window_token_pos for o in observations
                ],
            )

        else:
            token_observations = [
                embedded
                for embedded in observations
                if isinstance(embedded, EmbeddedObservation)
            ]

            insert_events(
                self.conn,
                event_id=[o.observation.event_id for o in observations],
                corpus=[o.observation.corpus for o in observations],
                doc_id=[o.observation.doc_id for o in observations],
                token=[o.observation.token for o in observations],
                token_idx=[o.observation.token_idx for o in observations],
                span_end_idx=[None for _ in observations],
                pub_year=[o.observation.pub_year for o in observations],
                local_window_id=[o.observation.local_window_id for o in observations],
                local_window_token_pos=[
                    o.observation.local_window_token_pos for o in observations
                ],
                medium_window_id=[o.observation.medium_window_id for o in observations],
                medium_window_token_pos=[
                    o.observation.medium_window_token_pos for o in observations
                ],
                broad_window_id=[o.observation.broad_window_id for o in observations],
                broad_window_token_pos=[
                    o.observation.broad_window_token_pos for o in observations
                ],
            )

        self.conn.commit()

    def _write_lance(
        self,
        observations: list[AnyEmbedded],
        *,
        scales: tuple[str, ...] = ACTIVE_SCALES,
    ) -> int:

        batches: dict[
            tuple[str, int],
            list[AnyEmbedded],
        ] = defaultdict(list)

        for embedded in observations:
            observation = embedded.observation

            if observation.event_id is None:
                raise RuntimeError(
                    "Cannot write an observation to Lance without "
                    "an authoritative PostgreSQL event ID."
                )

            if observation.pub_year is None:
                raise ValueError(
                    f"Observation {observation.event_id} has no "
                    "publication year and cannot be assigned to "
                    "a chronological Lance table."
                )

            for scale in scales:
                if scale not in embedded.vectors:
                    raise RuntimeError(
                        f"Observation {observation.event_id} has no "
                        f"vector for scale {scale!r}"
                    )

                batches[
                    (scale, observation.pub_year)
                ].append(embedded)

        total_written = 0

        for (scale, pub_year), batch in batches.items():
            writer = self._writer_for_scale(scale)

            event_ids = [
                embedded.observation.event_id
                for embedded in batch
            ]

            vectors = np.stack(
                [
                    embedded.vectors[scale]
                    for embedded in batch
                ]
            )

            total_written += writer.write(
                event_ids=event_ids,
                pub_year=pub_year,
                vectors=vectors,
            )

        return total_written

    def _resolve_repair_event_ids(
        self,
        observations: list[AnyEmbedded],
    ) -> None:
        """
        Resolve repair IDs using complete observation provenance.

        Token identity:
            corpus + doc_id + token_idx + window provenance

        Phrase identity:
            corpus + doc_id + token_idx + span_end_idx +
            window provenance
        """

        for embedded in observations:
            observation = embedded.observation

            if isinstance(
                observation,
                SpanObservation,
            ):
                clauses = [
                    "corpus = %s",
                    "doc_id = %s",
                    "token_idx = %s",
                    "span_end_idx = %s",
                    "local_window_id IS NOT DISTINCT FROM %s",
                    "local_window_token_pos IS NOT DISTINCT FROM %s",
                    "medium_window_id IS NOT DISTINCT FROM %s",
                    "medium_window_token_pos IS NOT DISTINCT FROM %s",
                    "broad_window_id IS NOT DISTINCT FROM %s",
                    "broad_window_token_pos IS NOT DISTINCT FROM %s",
                ]

                params = (
                    observation.corpus,
                    observation.doc_id,
                    observation.token_idx,
                    observation.span_end_idx,
                    observation.local_window_id,
                    observation.local_window_token_pos,
                    observation.medium_window_id,
                    observation.medium_window_token_pos,
                    observation.broad_window_id,
                    observation.broad_window_token_pos,
                )

            else:
                clauses = [
                    "corpus = %s",
                    "doc_id = %s",
                    "token_idx = %s",
                    "span_end_idx IS NULL",
                    "local_window_id IS NOT DISTINCT FROM %s",
                    "local_window_token_pos IS NOT DISTINCT FROM %s",
                    "medium_window_id IS NOT DISTINCT FROM %s",
                    "medium_window_token_pos IS NOT DISTINCT FROM %s",
                    "broad_window_id IS NOT DISTINCT FROM %s",
                    "broad_window_token_pos IS NOT DISTINCT FROM %s",
                ]

                params = (
                    observation.corpus,
                    observation.doc_id,
                    observation.token_idx,
                    observation.local_window_id,
                    observation.local_window_token_pos,
                    observation.medium_window_id,
                    observation.medium_window_token_pos,
                    observation.broad_window_id,
                    observation.broad_window_token_pos,
                )

            with self.conn.cursor() as cur:
                cur.execute(
                    f""" SELECT event_id FROM events WHERE {" AND ".join(clauses)} """,
                    params,
                )

                rows = cur.fetchall()

            if not rows:
                raise RuntimeError(
                    "Lance repair could not find an existing PostgreSQL "
                    "event for observation "
                    f"{observation.corpus}/"
                    f"{observation.doc_id}/"
                    f"{observation.token_idx}"
                )

            if len(rows) > 1:
                raise RuntimeError(
                    "Lance repair found multiple PostgreSQL events for "
                    "the same complete observation provenance: "
                    f"{observation.corpus}/"
                    f"{observation.doc_id}/"
                    f"{observation.token_idx}"
                )

            observation.event_id = int(rows[0][0])
