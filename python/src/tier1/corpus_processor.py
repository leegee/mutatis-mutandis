from lib.corpus_logging import logger

from tier1.models import *
from tier1.macberth_pipeline import MacBERThPipeline
from tier1.event_writer import EventWriter
from tier1.vector_writer import VectorWriter

class CorpusProcessor:
    def __init__(
        self,
        conn,
        pipeline: MacBERThPipeline | None,
        writer: EventWriter | None,
        *,
        neighbour_radius: int = 256,
        report_every: int = 25,
    ) -> None:
        self.conn = conn
        self.pipeline = pipeline
        self.writer = writer
        self.neighbour_radius = neighbour_radius
        self.report_every = report_every

    # Normal token processing
    def process(
        self,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
    ) -> None:
        documents = self._find_seed_documents(
            corpus=corpus,
            doc_id=doc_id,
        )

        logger.info( "[tier1] seed documents: %d", len(documents), )

        completed_docs_by_corpus: dict[str, set[str]] = {}

        for document_corpus, _document_id in documents:
            if document_corpus not in completed_docs_by_corpus:
                completed_docs_by_corpus[
                    document_corpus
                ] = existing_event_documents(
                    self.writer.conn,
                    corpus=document_corpus,
                    phrases=False,
                )

        for number, (
            document_corpus,
            document_id,
        ) in enumerate(
            documents,
            start=1,
        ):
            completed_docs = completed_docs_by_corpus[ document_corpus ]

            if document_id in completed_docs:
                logger.info(
                    "[tier1] skipping completed token document "
                    "%s/%s",
                    document_corpus,
                    document_id,
                )
                continue

            started = time.perf_counter()

            outcome = self._process_token_document(
                document_corpus,
                document_id,
            )

            if outcome.status != "processed":
                continue

            elapsed = time.perf_counter() - started

            logger.info(
                "[tier1] %3d/%-3d %-4s %-15s "
                "seeds=%3d observations=%5d "
                "written=%5d elapsed=%7.2fs",
                number,
                len(documents),
                document_corpus,
                document_id,
                outcome.targets,
                outcome.observations,
                outcome.written,
                elapsed,
            )

            completed_docs.add(document_id)

            if number % self.report_every == 0:
                logger.info( "[tier1] processed %d documents", number, )

    def _process_token_document(
        self,
        corpus: str,
        doc_id: str,
        *,
        dry_run: bool = False,
    ) -> DocumentOutcome:
        """
        Embed and write the token observations of one document.

        Shared by the direct loop (process) and the queue worker. With
        dry_run the embeddings are computed but nothing is written.
        """

        document = self._load_document( corpus, doc_id, )

        if document is None:
            return DocumentOutcome("missing")

        seed_positions = {
            position
            for position, row in enumerate(document.rows)
            if is_seed(row.token)
        }

        if not seed_positions:
            return DocumentOutcome("no_targets")

        target_positions = self._select_neighbours(
            seed_positions,
            document,
        )

        embeddings = self.pipeline.embed(
            document,
            target_positions,
        )

        observations = self._build_observations(
            document,
            target_positions,
            embeddings,
        )

        written = (
            0
            if dry_run
            else self.writer.write(
                observations,
            )
        )

        return DocumentOutcome(
            "processed",
            targets=len(seed_positions),
            observations=len(observations),
            written=written,
        )

    # Phrase processing
    def process_phrases(
        self,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
    ) -> None:

        documents = self._find_phrase_documents( corpus=corpus, doc_id=doc_id, )

        logger.info( "[tier1] phrase documents: %d", len(documents), )

        completed_docs_by_corpus: dict[str, set[str]] = {}

        for document_corpus, _document_id in documents:
            if document_corpus not in completed_docs_by_corpus:
                completed_docs_by_corpus[
                    document_corpus
                ] = existing_event_documents(
                    self.writer.conn,
                    corpus=document_corpus,
                    phrases=True,
                )

        for number, (
            document_corpus,
            document_id,
        ) in enumerate(
            documents,
            start=1,
        ):
            completed_docs = completed_docs_by_corpus[
                document_corpus
            ]

            if document_id in completed_docs:
                logger.info(
                    "[tier1] skipping completed phrase document "
                    "%s/%s",
                    document_corpus,
                    document_id,
                )
                continue

            started = time.perf_counter()

            outcome = self._process_phrase_document(
                document_corpus,
                document_id,
            )

            if outcome.status != "processed":
                continue

            elapsed = time.perf_counter() - started

            logger.info(
                "[tier1] phrase %3d/%-3d %-4s %-15s "
                "spans=%5d written=%5d elapsed=%7.2fs",
                number,
                len(documents),
                document_corpus,
                document_id,
                outcome.observations,
                outcome.written,
                elapsed,
            )

            completed_docs.add(document_id)

    def _process_phrase_document(
        self,
        corpus: str,
        doc_id: str,
        *,
        dry_run: bool = False,
    ) -> DocumentOutcome:
        """
        Embed and write the phrase observations of one document.

        Shared by the direct loop (process_phrases) and the queue worker.
        With dry_run the embeddings are computed but nothing is written.
        """

        document = self._load_document( corpus, doc_id, )

        if document is None:
            return DocumentOutcome("missing")

        spans = self._find_phrase_spans(
            document,
        )

        if not spans:
            return DocumentOutcome("no_targets")

        observations: list[EmbeddedSpanObservation] = []

        for (
            start_position,
            end_position,
            phrase,
        ) in spans:

            embeddings = self.pipeline.embed_span(
                document,
                start_position,
                end_position,
            )

            observations.append(
                self._build_span_observation(
                    document=document,
                    start_position=start_position,
                    end_position=end_position,
                    phrase=phrase,
                    embeddings=embeddings,
                )
            )

        written = (
            0
            if dry_run
            else self.writer.write(
                observations,
            )
        )

        return DocumentOutcome(
            "processed",
            targets=len(spans),
            observations=len(observations),
            written=written,
        )

    # Job queue
    def populate_jobs(
        self,
        kind: str,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> int:
        """
        Enqueue one job per document that still needs work of this kind.
        Safe to repeat: already-queued documents are left untouched.

        Only needs a database connection, so it can run without loading
        MacBERTh (pipeline and writer may be None for this call).
        """

        scale = backfill_scale_for_kind(kind)

        if scale is not None:
            documents = self._documents_missing_scale(
                scale,
                corpus,
                doc_id,
                min_year=min_year,
                max_year=max_year,
            )

        else:
            phrases = kind == JOB_KIND_PHRASES

            if phrases:
                candidates = self._find_phrase_documents(
                    corpus=corpus,
                    doc_id=doc_id,
                    min_year=min_year,
                    max_year=max_year,
                )
            else:
                candidates = self._find_seed_documents(
                    corpus=corpus,
                    doc_id=doc_id,
                    min_year=min_year,
                    max_year=max_year,
                )

            # Same completion rule as the direct loops: a document that
            # already has events of this type is not queued again.
            completed_by_corpus: dict[str, set[str]] = {}
            documents = []

            for document_corpus, document_id in candidates:
                if document_corpus not in completed_by_corpus:
                    completed_by_corpus[
                        document_corpus
                    ] = existing_event_documents(
                        self.conn,
                        corpus=document_corpus,
                        phrases=phrases,
                    )

                if document_id in completed_by_corpus[document_corpus]:
                    continue

                documents.append((document_corpus, document_id))

            logger.info(
                "[tier1] %d candidate documents, %d already complete",
                len(candidates),
                len(candidates) - len(documents),
            )

        return insert_jobs(
            self.conn,
            kind=kind,
            documents=documents,
        )

    def process_queue(
        self,
        *,
        kind: str,
        worker_id: str | None = None,
        max_docs: int | None = None,
        dry_run: bool = False,
        skip_indexing: bool = False,
    ) -> int:
        """
        Claim and process jobs of one kind until none are pending (or
        max_docs is reached). Returns the number of jobs attempted.

        A failed job is recorded as 'failed' with its error and the worker
        carries on with the next one.
        """

        worker_id = worker_id or _default_worker_id()
        scale = backfill_scale_for_kind(kind)

        vector_writer = (
            self.writer._writer_for_scale(scale)
            if scale is not None
            else None
        )

        logger.info(
            "[tier1] worker %s starting kind=%s (dry_run=%s)",
            worker_id,
            kind,
            dry_run,
        )

        processed = 0

        while True:
            if max_docs is not None and processed >= max_docs:
                break

            job = claim_job(
                self.conn,
                worker_id,
                kind=kind,
                dry_run=dry_run,
            )

            if job is None:
                logger.info(
                    "[tier1] no more pending %s jobs",
                    kind,
                )
                break

            job_id, job_corpus, job_doc_id = job
            started = time.perf_counter()

            try:
                outcome = self._run_job(
                    kind,
                    job_corpus,
                    job_doc_id,
                    vector_writer=vector_writer,
                    dry_run=dry_run,
                )

                if outcome.status == "missing":
                    mark_job_failed(
                        self.conn,
                        job_id,
                        "document not found in "
                        "pamphlet_tokens/pamphlet_corpus",
                        dry_run=dry_run,
                    )
                else:
                    mark_job_done(
                        self.conn,
                        job_id,
                        dry_run=dry_run,
                    )

                logger.info(
                    "[tier1] %s %-4s %-15s kind=%s status=%s "
                    "targets=%5d observations=%5d written=%5d "
                    "elapsed=%7.2fs",
                    "dry-run" if dry_run else "job",
                    job_corpus,
                    job_doc_id,
                    kind,
                    outcome.status,
                    outcome.targets,
                    outcome.observations,
                    outcome.written,
                    time.perf_counter() - started,
                )

            except Exception as exc:
                logger.exception(
                    "[tier1] failed %s/%s (kind=%s)",
                    job_corpus,
                    job_doc_id,
                    kind,
                )

                try:
                    self.conn.rollback()  # clear the aborted transaction
                except Exception:
                    pass

                mark_job_failed(
                    self.conn,
                    job_id,
                    str(exc),
                    dry_run=dry_run,
                )

            processed += 1

            # In dry-run we only ever look at the first pending job
            # (otherwise we would loop forever on the same row).
            if dry_run:
                break

        if vector_writer is not None and not (skip_indexing or dry_run):
            # Same finishing step backfill_scale() performs.
            vector_writer.index_existing_tables()

        logger.info(
            "[tier1] worker %s finished kind=%s (%d jobs)",
            worker_id,
            kind,
            processed,
        )

        return processed

    def _run_job(
        self,
        kind: str,
        corpus: str,
        doc_id: str,
        *,
        vector_writer: VectorWriter | None,
        dry_run: bool,
    ) -> DocumentOutcome:
        scale = backfill_scale_for_kind(kind)

        if scale is not None:
            # Someone may have backfilled this document since it was queued.
            if not document_missing_scale(
                self.conn,
                corpus=corpus,
                doc_id=doc_id,
                scale=scale,
            ):
                return DocumentOutcome("already_complete")

            if dry_run:
                return DocumentOutcome("would_backfill")

            return DocumentOutcome(
                "processed",
                written=self._backfill_document(
                    corpus,
                    doc_id,
                    scale,
                    vector_writer,
                ),
            )

        phrases = kind == JOB_KIND_PHRASES

        # Same safeguard as the direct loops: never write a second set of
        # events for a document that already has them (e.g. after a direct
        # run, or a worker that died after committing but before marking
        # its job done).
        if document_has_events(
            self.conn,
            corpus=corpus,
            doc_id=doc_id,
            phrases=phrases,
        ):
            logger.info(
                "[tier1] skipping completed %s document %s/%s",
                "phrase" if phrases else "token",
                corpus,
                doc_id,
            )

            return DocumentOutcome("already_complete")

        if phrases:
            return self._process_phrase_document(
                corpus,
                doc_id,
                dry_run=dry_run,
            )

        return self._process_token_document(
            corpus,
            doc_id,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair(
        self,
        *,
        corpus: str,
        doc_id: str,
        phrases: bool = False,
    ) -> None:

        started = time.perf_counter()

        logger.info(
            "[tier1] repair: %s/%s phrases=%s",
            corpus,
            doc_id,
            phrases,
        )

        document = self._load_document(
            corpus,
            doc_id,
        )

        if document is None:
            raise RuntimeError(
                f"Document not found: {corpus}/{doc_id}"
            )

        if phrases:
            existing = self._load_phrase_events(
                corpus,
                doc_id,
            )

            spans = self._find_phrase_spans(
                document,
            )

            current = {
                (
                    start,
                    end,
                )
                for start, end, _phrase in spans
            }

            stored = {
                (
                    start,
                    end,
                )
                for start, end, _event_id in existing
            }

            if current != stored:
                raise RuntimeError(
                    f"{corpus}/{doc_id}: phrase observation set "
                    "changed since original run "
                    f"(only_current={len(current - stored)}, "
                    f"only_stored={len(stored - current)})."
                )

            observations = []

            for (
                start,
                end,
                phrase,
            ) in spans:

                embeddings = self.pipeline.embed_span(
                    document,
                    start,
                    end,
                )

                observations.append(
                    self._build_span_observation(
                        document=document,
                        start_position=start,
                        end_position=end,
                        phrase=phrase,
                        embeddings=embeddings,
                    )
                )

        else:
            existing = self._load_token_events(
                corpus,
                doc_id,
            )

            seed_positions = {
                position
                for position, row in enumerate(document.rows)
                if is_seed(row.token)
            }

            if not seed_positions:
                raise RuntimeError(
                    f"No seed occurrences found: "
                    f"{corpus}/{doc_id}"
                )

            target_positions = self._select_neighbours(
                seed_positions,
                document,
            )

            expected = {
                document.rows[position].token_idx
                for position in target_positions
            }

            stored = set(existing)

            if expected != stored:
                raise RuntimeError(
                    f"{corpus}/{doc_id}: observation set changed "
                    "since original run "
                    f"(only_expected={len(expected - stored)}, "
                    f"only_stored={len(stored - expected)}). "
                    "Check CONCEPT_SETS / --neighbour-radius."
                )

            embeddings = self.pipeline.embed(
                document,
                target_positions,
            )

            observations = self._build_observations(
                document,
                target_positions,
                embeddings,
            )

        written = self.writer.repair_lance(
            observations,
        )

        elapsed = time.perf_counter() - started

        logger.info(
            "[tier1] repair complete: %-4s %-15s "
            "observations=%5d lance_written=%5d "
            "elapsed=%7.2fs",
            corpus,
            doc_id,
            len(observations),
            written,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Scale backfill
    # ------------------------------------------------------------------

    def backfill_scale(
        self,
        scale: str,
        *,
        corpus: str | None = None,
        doc_id: str | None = None,
    ) -> int:

        if scale not in SCALE_NAMES:
            raise ValueError(
                f"unknown scale {scale!r}"
            )

        writer = self.writer._writer_for_scale(
            scale,
        )

        documents = self._documents_missing_scale(
            scale,
            corpus,
            doc_id,
        )

        logger.info(
            "[backfill:%s] documents to process: %d",
            scale,
            len(documents),
        )

        for number, (
            doc_corpus,
            did,
        ) in enumerate(
            documents,
            start=1,
        ):
            started = time.perf_counter()

            n = self._backfill_document(
                doc_corpus,
                did,
                scale,
                writer,
            )

            logger.info(
                "[backfill:%s] %d/%d %s/%s "
                "events=%d elapsed=%.2fs",
                scale,
                number,
                len(documents),
                doc_corpus,
                did,
                n,
                time.perf_counter() - started,
            )

        writer.index_existing_tables()

        return len(documents)

    def _documents_missing_scale(
        self,
        scale: str,
        corpus: str | None,
        doc_id: str | None,
        *,
        min_year: int | None = None,
        max_year: int | None = None,
    ):
        clauses = [
            f"e.{scale}_window_id IS NULL",
        ]

        params: list[object] = []

        if corpus is not None:
            clauses.append(
                "e.corpus = %s"
            )
            params.append(corpus)

        if doc_id is not None:
            clauses.append(
                "e.doc_id = %s"
            )
            params.append(doc_id)

        if min_year is not None:
            clauses.append(
                "e.pub_year >= %s"
            )
            params.append(min_year)

        if max_year is not None:
            clauses.append(
                "e.pub_year <= %s"
            )
            params.append(max_year)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT e.corpus, e.doc_id
                FROM events AS e
                WHERE {" AND ".join(clauses)}
                ORDER BY e.corpus, e.doc_id
                """,
                params,
            )

            return cur.fetchall()

    def _backfill_document(
        self,
        corpus: str,
        doc_id: str,
        scale: str,
        vector_writer: VectorWriter,
    ) -> int:

        document = self._load_document(
            corpus,
            doc_id,
        )

        if document is None:
            raise RuntimeError(
                f"Document not found: {corpus}/{doc_id}"
            )

        token_events = self._load_token_events(
            corpus,
            doc_id,
        )

        phrase_events = self._load_phrase_events(
            corpus,
            doc_id,
        )

        total = 0

        if token_events:
            total += self._backfill_token_events(
                document,
                token_events,
                scale,
                vector_writer,
            )

        if phrase_events:
            total += self._backfill_phrase_events(
                document,
                phrase_events,
                scale,
                vector_writer,
            )

        return total

    def _backfill_token_events(
        self,
        document: DocBuffer,
        events: dict[int, int],
        scale: str,
        vector_writer: VectorWriter,
    ) -> int:

        seed_positions = {
            position
            for position, row in enumerate(document.rows)
            if is_seed(row.token)
        }

        targets = self._select_neighbours(
            seed_positions,
            document,
        )

        expected = {
            document.rows[position].token_idx
            for position in targets
        }

        stored = set(events)

        if expected != stored:
            raise RuntimeError(
                f"{document.corpus}/{document.doc_id}: token "
                "observation set changed since original run "
                f"(only_expected={len(expected - stored)}, "
                f"only_stored={len(stored - expected)})."
            )

        embeddings = self.pipeline.embed(
            document,
            targets,
            scales=(scale,),
        )

        by_year: dict[
            int,
            tuple[list[int], list[np.ndarray]],
        ] = defaultdict(
            lambda: ([], [])
        )

        updates: list[
            tuple[int, int, int]
        ] = []

        for position in sorted(targets):
            row = document.rows[position]

            event_id = events[row.token_idx]

            if row.pub_year is None:
                raise RuntimeError(
                    f"{document.corpus}/{document.doc_id}: "
                    f"event {event_id} has no publication year."
                )

            embedded = embeddings[position][scale]

            ids, vectors = by_year[row.pub_year]

            ids.append(event_id)
            vectors.append(embedded.vector)

            updates.append(
                (
                    embedded.window_id,
                    embedded.window_token_pos,
                    event_id,
                )
            )

        for pub_year, (
            ids,
            vectors,
        ) in by_year.items():

            vector_writer.write(
                event_ids=ids,
                pub_year=pub_year,
                vectors=np.stack(vectors),
            )

        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                UPDATE events
                SET {scale}_window_id = %s,
                    {scale}_window_token_pos = %s
                WHERE event_id = %s
                """,
                updates,
            )

        self.conn.commit()

        return len(updates)

    def _backfill_phrase_events(
        self,
        document: DocBuffer,
        events: list[
            tuple[int, int, int]
        ],
        scale: str,
        vector_writer: VectorWriter,
    ) -> int:

        current_spans = {
            (
                start,
                end,
            ): phrase
            for start, end, phrase
            in self._find_phrase_spans(document)
        }

        stored_spans = {
            (
                start,
                end,
            )
            for start, end, _event_id
            in events
        }

        if set(current_spans) != stored_spans:
            raise RuntimeError(
                f"{document.corpus}/{document.doc_id}: phrase "
                "observation set changed since original run."
            )

        by_year: dict[
            int,
            tuple[list[int], list[np.ndarray]],
        ] = defaultdict(
            lambda: ([], [])
        )

        updates: list[
            tuple[int, int, int]
        ] = []

        for (
            start,
            end,
            event_id,
        ) in events:

            embeddings = self.pipeline.embed_span(
                document,
                start,
                end,
                scales=(scale,),
            )

            embedded = embeddings[scale]

            pub_year = document.pub_year

            if pub_year is None:
                raise RuntimeError(
                    f"{document.corpus}/{document.doc_id}: "
                    f"phrase event {event_id} has no publication year."
                )

            ids, vectors = by_year[pub_year]

            ids.append(event_id)
            vectors.append(embedded.vector)

            updates.append(
                (
                    embedded.window_id,
                    embedded.window_token_pos,
                    event_id,
                )
            )

        for pub_year, (
            ids,
            vectors,
        ) in by_year.items():

            vector_writer.write(
                event_ids=ids,
                pub_year=pub_year,
                vectors=np.stack(vectors),
            )

        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                UPDATE events
                SET {scale}_window_id = %s,
                    {scale}_window_token_pos = %s
                WHERE event_id = %s
                """,
                updates,
            )

        self.conn.commit()

        return len(updates)

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _build_observations(
        self,
        document: DocBuffer,
        target_positions: set[int],
        embeddings: dict[ int, dict[str, EmbeddedVector], ],
    ) -> list[EmbeddedObservation]:

        observations: list[ EmbeddedObservation ] = []

        for position in sorted(target_positions):
            row = document.rows[position]

            vectors: dict[str, np.ndarray] = {}
            provenance: dict[ str, EmbeddedVector, ] = {}

            for scale in ACTIVE_SCALES:
                embedded = embeddings[position].get( scale )

                if embedded is None:
                    raise RuntimeError(
                        "Missing embedding provenance for "
                        f"{document.corpus}/{document.doc_id}/"
                        f"{row.token_idx}, scale={scale}"
                    )

                vectors[scale] = embedded.vector
                provenance[scale] = embedded

            local = provenance.get("local")
            medium = provenance.get("medium")
            broad = provenance.get("broad")

            observation = Observation(
                event_id=None,
                corpus=row.corpus,
                doc_id=row.doc_id,
                token=row.token,
                token_idx=row.token_idx,
                pub_year=row.pub_year,

                local_window_id=(
                    local.window_id
                    if local is not None
                    else None
                ),
                local_window_token_pos=(
                    local.window_token_pos
                    if local is not None
                    else None
                ),

                medium_window_id=(
                    medium.window_id
                    if medium is not None
                    else None
                ),
                medium_window_token_pos=(
                    medium.window_token_pos
                    if medium is not None
                    else None
                ),

                broad_window_id=(
                    broad.window_id
                    if broad is not None
                    else None
                ),
                broad_window_token_pos=(
                    broad.window_token_pos
                    if broad is not None
                    else None
                ),
            )

            observations.append(
                EmbeddedObservation(
                    observation=observation,
                    vectors=vectors,
                )
            )

        return observations

    def _build_span_observation(
        self,
        *,
        document: DocBuffer,
        start_position: int,
        end_position: int,
        phrase: str,
        embeddings: dict[
            str,
            EmbeddedVector,
        ],
    ) -> EmbeddedSpanObservation:

        start_row = document.rows[
            start_position
        ]

        vectors = {
            scale: embeddings[scale].vector
            for scale in ACTIVE_SCALES
        }

        local = embeddings.get("local")
        medium = embeddings.get("medium")
        broad = embeddings.get("broad")

        observation = SpanObservation(
            event_id=None,
            corpus=document.corpus,
            doc_id=document.doc_id,
            phrase=phrase,
            token_idx=start_row.token_idx,
            span_end_idx=document.rows[
                end_position - 1
            ].token_idx,
            pub_year=document.pub_year,

            local_window_id=(
                local.window_id
                if local is not None
                else None
            ),
            local_window_token_pos=(
                local.window_token_pos
                if local is not None
                else None
            ),

            medium_window_id=(
                medium.window_id
                if medium is not None
                else None
            ),
            medium_window_token_pos=(
                medium.window_token_pos
                if medium is not None
                else None
            ),

            broad_window_id=(
                broad.window_id
                if broad is not None
                else None
            ),
            broad_window_token_pos=(
                broad.window_token_pos
                if broad is not None
                else None
            ),
        )

        return EmbeddedSpanObservation(
            observation=observation,
            vectors=vectors,
        )

    # ------------------------------------------------------------------
    # Corpus lookup
    # ------------------------------------------------------------------

    def _find_seed_documents(
        self,
        *,
        corpus: str | None,
        doc_id: str | None,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> list[tuple[str, str]]:

        clauses = [
            "lower(t.token) = ANY(%s)",
        ]

        params: list[object] = [
            sorted(
                SEED_FORMS - FALSE_POSITIVE_FORMS
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

        join = ""

        if min_year is not None or max_year is not None:
            join = (
                "JOIN pamphlet_corpus AS d "
                "ON d.corpus = t.corpus AND d.doc_id = t.doc_id"
            )

            if min_year is not None:
                clauses.append("d.pub_year >= %s")
                params.append(min_year)

            if max_year is not None:
                clauses.append("d.pub_year <= %s")
                params.append(max_year)

        sql = f"""
            SELECT DISTINCT t.corpus, t.doc_id
            FROM pamphlet_tokens AS t
            {join}
            WHERE {" AND ".join(clauses)}
            ORDER BY t.corpus, t.doc_id
        """

        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                params,
            )

            return cur.fetchall()

    def _find_phrase_documents(
        self,
        *,
        corpus: str | None,
        doc_id: str | None,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> list[tuple[str, str]]:

        if not PHRASE_FORMS:
            return []

        first_tokens = sorted(
            {
                phrase[0]
                for phrase in PHRASE_FORMS
                if phrase
            }
        )

        clauses = [
            "lower(t.token) = ANY(%s)",
        ]

        params: list[object] = [
            first_tokens,
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

        join = ""

        if min_year is not None or max_year is not None:
            join = (
                "JOIN pamphlet_corpus AS d "
                "ON d.corpus = t.corpus AND d.doc_id = t.doc_id"
            )

            if min_year is not None:
                clauses.append("d.pub_year >= %s")
                params.append(min_year)

            if max_year is not None:
                clauses.append("d.pub_year <= %s")
                params.append(max_year)

        sql = f"""
            SELECT DISTINCT t.corpus, t.doc_id
            FROM pamphlet_tokens AS t
            {join}
            WHERE {" AND ".join(clauses)}
            ORDER BY t.corpus, t.doc_id
        """

        with self.conn.cursor() as cur:
            cur.execute(
                sql,
                params,
            )

            return cur.fetchall()


    def _load_document(
        self,
        corpus: str,
        doc_id: str,
    ) -> DocBuffer | None:

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.corpus, t.doc_id, t.token_idx, t.token, d.pub_year
                FROM pamphlet_tokens AS t
                JOIN pamphlet_corpus AS d
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

            token_rows = [
                TokenRow(
                    corpus=row[0],
                    doc_id=row[1],
                    token_idx=row[2],
                    token=row[3],
                    pub_year=row[4],
                )
                for row in rows
            ]

            for row in token_rows:
                if row.doc_id == "A01289" and row.token.lower() == "clotho":
                    logger.debug("DEBUG TOKEN ROW: %r", row)

            return DocBuffer(
                corpus=corpus,
                doc_id=doc_id,
                pub_year=token_rows[0].pub_year,
                rows=token_rows,
            )


    # ------------------------------------------------------------------
    # Phrase matching
    # ------------------------------------------------------------------

    def _find_phrase_spans(
        self,
        document: DocBuffer,
    ) -> list[
        tuple[int, int, str]
    ]:

        normalised = [
            normalise_token(row.token)
            for row in document.rows
        ]

        spans: list[
            tuple[int, int, str]
        ] = []

        for phrase in PHRASE_FORMS:
            length = len(phrase)

            if length == 0:
                continue

            for start in range(
                0,
                len(normalised) - length + 1,
            ):
                if tuple(
                    normalised[
                        start:start + length
                    ]
                ) != phrase:
                    continue

                end = start + length

                spans.append(
                    (
                        start,
                        end,
                        " ".join(phrase),
                    )
                )

        spans.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        return spans

    # ------------------------------------------------------------------
    # Neighbours
    # ------------------------------------------------------------------

    def _select_neighbours(
        self,
        seed_positions: set[int],
        document: DocBuffer,
    ) -> set[int]:

        positions: set[int] = set()

        for seed_position in seed_positions:
            start = max(
                0,
                seed_position - self.neighbour_radius,
            )

            end = min(
                len(document.rows),
                seed_position + self.neighbour_radius + 1,
            )

            positions.update(
                position
                for position in range(
                    start,
                    end,
                )
                if (
                    position in seed_positions
                    or is_storable_event(
                        document.rows[position].token
                    )
                )
            )

        return positions

    # ------------------------------------------------------------------
    # Existing event loading
    # ------------------------------------------------------------------

    def _load_token_events(
        self,
        corpus: str,
        doc_id: str,
    ) -> dict[int, int]:

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    token_idx,
                    event_id
                FROM events
                WHERE corpus = %s
                  AND doc_id = %s
                  AND span_end_idx IS NULL
                ORDER BY token_idx
                """,
                (
                    corpus,
                    doc_id,
                ),
            )

            rows = cur.fetchall()

        events: dict[int, int] = {}

        for token_idx, event_id in rows:
            if token_idx in events:
                raise RuntimeError(
                    f"{corpus}/{doc_id}: multiple token events "
                    f"at token_idx={token_idx}"
                )

            events[int(token_idx)] = int(event_id)

        return events

    def _load_phrase_events(
        self,
        corpus: str,
        doc_id: str,
    ) -> list[
        tuple[int, int, int]
    ]:

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    token_idx,
                    span_end_idx,
                    event_id
                FROM events
                WHERE corpus = %s
                  AND doc_id = %s
                  AND span_end_idx IS NOT NULL
                ORDER BY token_idx, span_end_idx
                """,
                (
                    corpus,
                    doc_id,
                ),
            )

            rows = cur.fetchall()

        return [
            (
                int(token_idx),
                int(span_end_idx),
                int(event_id),
            )
            for token_idx, span_end_idx, event_id
            in rows
        ]
