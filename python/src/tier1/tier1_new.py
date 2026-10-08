# tier1/tier1_new.py
"""
Build Tier 1 token and phrase observations from the restricted pamphlet
corpus through MacBERTh into Lance.

Two ways to run it:

1. Direct (unchanged): iterate documents in-process, skipping any that
   already have events of the requested type.

       python tier1_new.py [--corpus C] [--doc-id D] [--phrases]
       python tier1_new.py --add-scale medium
       python tier1_new.py --repair CORPUS/DOC_ID
       ...

2. Job queue: any number of workers claim documents from the
   `embedding_jobs` table using FOR UPDATE SKIP LOCKED, so they can run
   side by side without coordination. A job is one (corpus, doc_id, kind);
   `kind` is one of

       tokens            token observations          (default)
       phrases           phrase observations         (--phrases)
       backfill:<scale>  add a scale to existing     (--add-scale <scale>)
                         events, no new event IDs

   Token and phrase passes (and each backfill scale) are independent, so
   queueing one never marks another done.

       # 1. enqueue (idempotent; add --phrases / --add-scale to pick a kind)
       python tier1_new.py --populate [--corpus C] \\
              [--min-year Y] [--max-year Y]

       # 2. run one or more workers
       python tier1_new.py --worker [--worker-id W] \\
              [--max-docs N] [--dry-run] [--skip-indexing]

   With several workers, pass --skip-indexing to each and finish with a
   single `--index-only` run, rather than having every worker rebuild
   the indexes.

   --dry-run claims nothing, writes nothing and leaves jobs untouched: it
   embeds the first pending job so you can measure time / catch OOM.

Reset stuck or failed jobs of one kind:

    UPDATE embedding_jobs
    SET status      = 'pending',
        worker_id   = NULL,
        claimed_at  = NULL,
        finished_at = NULL,
        error       = NULL
    WHERE status IN ('running', 'failed')
      AND kind = 'tokens';
"""

from __future__ import annotations

import argparse
import os
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lib.corpus_config import (
    ACTIVE_SCALES,
    CONCEPT_SETS,
    PHRASE_SETS,
    EMBED_BATCH_SIZE,
    LANCE_INDEXES_DIR,
    WINDOW_CONFIGS, SCALE_NAMES, LANCE_MODEL_NAME,LANCE_BUCKET_SIZE
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import load_macberth
from lib.stopwords_min import STOPWORDS

from tier1.models import *
from tier1.db_observation_backend import (
    allocate_event_ids,
    insert_events,
    create_events_table,
)
from tier1.vector_writer import VectorWriter
from tier1.event_writer import EventWriter

from tier1.doc_buffer import DocBuffer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

# Token helpers
def normalise_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).strip().lower()


def is_punctuation(token: str) -> bool:
    value = normalise_token(token)

    return bool(value) and all(
        unicodedata.category(char).startswith("P")
        for char in value
    )


def is_stopword(token: str) -> bool:
    return normalise_token(token) in STOPWORDS


def is_storable_event(token: str) -> bool:
    return not is_stopword(token) and not is_punctuation(token)


# Seed / phrase forms
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


def phrase_forms() -> list[tuple[str, ...]]:
    forms: list[tuple[str, ...]] = []

    for phrase_set in PHRASE_SETS.values():
        for phrase in phrase_set:
            if isinstance(phrase, str):
                tokens = tuple(
                    normalise_token(token)
                    for token in phrase.split()
                    if token.strip()
                )
            else:
                tokens = tuple(
                    normalise_token(token)
                    for token in phrase
                )

            if tokens:
                forms.append(tokens)

    return forms


PHRASE_FORMS = phrase_forms()


# Existing-event helpers
def existing_event_documents(
    conn,
    *,
    corpus: str,
    phrases: bool = False,
) -> set[str]:
    """
    Return documents which already have observations of the requested type.

    Token observations:
        span_end_idx IS NULL

    Phrase observations:
        span_end_idx IS NOT NULL

    This deliberately keeps token and phrase completion independent.
    """

    span_clause = (
        "e.span_end_idx IS NOT NULL"
        if phrases
        else "e.span_end_idx IS NULL"
    )

    with conn.cursor() as cur:
        cur.execute(
            f""" SELECT DISTINCT e.doc_id FROM events AS e WHERE e.corpus = %s AND {span_clause} """,
            (corpus,),
        )

        return {
            row[0]
            for row in cur.fetchall()
        }


def document_has_events(
    conn,
    *,
    corpus: str,
    doc_id: str,
    phrases: bool = False,
) -> bool:
    """
    Single-document form of existing_event_documents(): does this document
    already have observations of the requested type?
    """

    span_clause = (
        "span_end_idx IS NOT NULL"
        if phrases
        else "span_end_idx IS NULL"
    )

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT EXISTS (
                SELECT 1 FROM events
                WHERE corpus = %s AND doc_id = %s AND {span_clause} )
            """,
            (corpus, doc_id),
        )

        return bool(cur.fetchone()[0])


def document_missing_scale(
    conn,
    *,
    corpus: str,
    doc_id: str,
    scale: str,
) -> bool:
    """
    Does this document still have events with no `scale` window
    provenance (i.e. is there anything left to backfill)?
    """

    if scale not in SCALE_NAMES:
        raise ValueError(f"unknown scale {scale!r}")

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT EXISTS (
                SELECT 1 FROM events WHERE corpus = %s AND doc_id = %s AND {scale}_window_id IS NULL
            )
            """,
            (corpus, doc_id),
        )

        return bool(cur.fetchone()[0])


# Job queue
#
# Any number of workers (local or remote) can run at once: work is claimed
# with FOR UPDATE SKIP LOCKED, filtered to one job kind per worker.
#
# This is a table of its own (not corpus2events' `embedding_jobs`): that
# table is keyed on (corpus, doc_id, scale) for window passes, whereas
# here the unit of work is (corpus, doc_id, kind) and a token pass, a
# phrase pass and each scale backfill must complete independently.

JOBS_TABLE = "embedding_jobs"

JOB_KIND_TOKENS = "tokens"
JOB_KIND_PHRASES = "phrases"
JOB_KIND_BACKFILL_PREFIX = "backfill:"


def job_kind(
    *,
    phrases: bool = False,
    add_scale: str | None = None,
) -> str:
    """
    Map the CLI flags to a job kind, mirroring main()'s dispatch order:
    --add-scale wins over --phrases (backfill covers token and phrase
    events alike).
    """

    if add_scale is not None:
        if add_scale not in SCALE_NAMES:
            raise ValueError(f"unknown scale {add_scale!r}")

        return f"{JOB_KIND_BACKFILL_PREFIX}{add_scale}"

    return JOB_KIND_PHRASES if phrases else JOB_KIND_TOKENS


def backfill_scale_for_kind(kind: str) -> str | None:
    """Return the scale of a `backfill:<scale>` kind, else None."""

    if not kind.startswith(JOB_KIND_BACKFILL_PREFIX):
        return None

    scale = kind[len(JOB_KIND_BACKFILL_PREFIX):]

    if scale not in SCALE_NAMES:
        raise ValueError(f"unknown scale in job kind {kind!r}")

    return scale


def _default_worker_id() -> str:
    # os.uname() does not exist on Windows.
    try:
        host = os.uname().nodename
    except AttributeError:
        host = os.environ.get("COMPUTERNAME", "windows")

    return f"{host}-{os.getpid()}"


def ensure_jobs_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {JOBS_TABLE} (
                job_id      BIGSERIAL PRIMARY KEY,
                corpus      TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                kind        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                worker_id   TEXT,
                claimed_at  TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                error       TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_{JOBS_TABLE}_corpus_doc_kind
                ON {JOBS_TABLE} (corpus, doc_id, kind);
            CREATE INDEX IF NOT EXISTS idx_{JOBS_TABLE}_status_kind
                ON {JOBS_TABLE} (status, kind);
            """
        )

    conn.commit()


def insert_jobs(
    conn,
    *,
    kind: str,
    documents: list[tuple[str, str]],
) -> int:
    """
    Insert one (corpus, doc_id, kind) job per document not already
    queued. Returns the number of jobs actually created.
    """

    if not documents:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {JOBS_TABLE} (corpus, doc_id, kind)
            SELECT u.corpus, u.doc_id, %s
            FROM unnest(%s::text[], %s::text[]) AS u(corpus, doc_id)
            ON CONFLICT (corpus, doc_id, kind) DO NOTHING
            """,
            (
                kind,
                [corpus for corpus, _doc_id in documents],
                [doc_id for _corpus, doc_id in documents],
            ),
        )

        created = cur.rowcount

    conn.commit()

    return created


def claim_job(
    conn,
    worker_id: str,
    *,
    kind: str,
    dry_run: bool = False,
) -> tuple[int, str, str] | None:
    """Claim one pending job of this kind. Returns (job_id, corpus, doc_id) or None."""

    if dry_run:
        # See real pending work, but never mark it running.
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT job_id, corpus, doc_id
                FROM {JOBS_TABLE}
                WHERE status = 'pending' AND kind = %s
                ORDER BY job_id
                LIMIT 1
                """,
                (kind,),
            )

            return cur.fetchone()

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {JOBS_TABLE}
            SET status = 'running',
                worker_id = %s,
                claimed_at = now()
            WHERE job_id = (
                SELECT job_id
                FROM {JOBS_TABLE}
                WHERE status = 'pending' AND kind = %s
                ORDER BY job_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING job_id, corpus, doc_id
            """,
            (worker_id, kind),
        )

        row = cur.fetchone()

    conn.commit()

    return row


def mark_job_done(
    conn,
    job_id: int,
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {JOBS_TABLE}
            SET status = 'done', finished_at = now(), error = NULL
            WHERE job_id = %s
            """,
            (job_id,),
        )

    conn.commit()


def mark_job_failed(
    conn,
    job_id: int,
    error: str,
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {JOBS_TABLE}
            SET status = 'failed', finished_at = now(), error = %s
            WHERE job_id = %s
            """,
            (error[:2000], job_id),
        )

    conn.commit()


# CLI helpers

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


def repair_year_range(
    processor: CorpusProcessor,
    conn,
    start_year: int,
    end_year: int,
    corpus: str | None = None,
) -> int:

    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) must not exceed "
            f"end_year ({end_year})"
        )

    bucket_start = (
        start_year // LANCE_BUCKET_SIZE
    ) * LANCE_BUCKET_SIZE

    bucket_end = (
        (end_year // LANCE_BUCKET_SIZE) + 1
    ) * LANCE_BUCKET_SIZE - 1

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT e.corpus, e.doc_id, e.pub_year
            FROM events AS e
            WHERE (%s::text IS NULL OR e.corpus = %s)
              AND e.pub_year BETWEEN %s AND %s
            ORDER BY
                e.pub_year,
                e.corpus,
                e.doc_id
            """,
            (
                corpus,
                corpus,
                bucket_start,
                bucket_end,
            ),
        )

        documents = cur.fetchall()

    logger.info(
        "[repair] Requested years %d-%d; repairing complete "
        "buckets %d-%d; %d documents "
        "(corpus=%s)",
        start_year,
        end_year,
        bucket_start,
        bucket_end,
        len(documents),
        corpus or "all",
    )

    repaired = 0

    for number, (
        doc_corpus,
        doc_id,
        pub_year,
    ) in enumerate(
        documents,
        start=1,
    ):
        logger.info(
            "[repair] %d/%d %s/%s "
            "(pub_year=%s)",
            number,
            len(documents),
            doc_corpus,
            doc_id,
            pub_year,
        )

        processor.repair(
            corpus=doc_corpus,
            doc_id=doc_id,
            phrases=False,
        )

        repaired += 1

    results = processor.writer.purge_orphans(
        conn,
        year_range=(
            bucket_start,
            bucket_end,
        ),
        apply=True,
    )

    logger.info(
        "[repair] purged orphans: %s",
        results,
    )

    logger.info(
        "[repair] Repaired %d documents "
        "for buckets %d-%d",
        repaired,
        bucket_start,
        bucket_end,
    )

    return repaired


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=( "Build Tier 1 token and phrase observations from the restricted pamphlet corpus through MacBERTh into Lance." )
    )

    parser.add_argument( "--corpus", default=None, )

    parser.add_argument( "--doc-id", default=None, )

    parser.add_argument( "--neighbour-radius", type=int, default=256, )

    parser.add_argument( "--lance-root", type=Path, default=Path(LANCE_INDEXES_DIR), )

    parser.add_argument( "--batch-size", type=int, default=EMBED_BATCH_SIZE, )

    parser.add_argument( "--report-every", type=int, default=1, )

    parser.add_argument( "--phrases", action="store_true", help="Process phrase observations rather than token observations.", )

    parser.add_argument( "--add-scale", choices=SCALE_NAMES, default=None, help=( "Add this scale to existing observations without creating new event IDs." ), )

    parser.add_argument( "--mask", action="store_true", help=( "Replace target tokens with [MASK] before embedding." ), )

    parser.add_argument( "--index-only", action="store_true", help=( "Rebuild incomplete indexes on existing active-scale Lance tables." ), )

    parser.add_argument(
        "--repair",
        type=parse_repair_target,
        metavar="CORPUS/DOC_ID",
        help=( "Regenerate Lance vectors for one token-observation document without modifying PostgreSQL events." ),
    )

    parser.add_argument(
        "--repair-years",
        nargs=2,
        type=int,
        metavar=("START_YEAR", "END_YEAR"),
        help=( "Repair all token-observation documents in the complete 50-year Lance buckets containing this range." ),
    )

    parser.add_argument( "--skip-indexing", action="store_true", help=( "Skip the post-run Lance index rebuild." ), )

    # -- Job queue ---------------------------------------------------------

    parser.add_argument( "--populate", action="store_true",
        help=( "Enqueue jobs (one per document) instead of processing. Combine with --phrases or --add-scale to pick the job kind; --corpus, --doc-id, --min-year and --max-year narrow it." ),
    )

    parser.add_argument( "--worker", action="store_true",
        help=( "Claim and process queued jobs until none remain. Combine with --phrases or --add-scale to pick the job kind." ),
    )

    parser.add_argument( "--min-year", type=int, default=None,
        help="With --populate: only documents published in/after this year.",
    )

    parser.add_argument( "--max-year", type=int, default=None,
        help="With --populate: only documents published in/before this year.",
    )

    parser.add_argument( "--worker-id", default=None, help="With --worker: identifier recorded on claimed jobs.", )

    parser.add_argument( "--max-docs", type=int, default=None,
        help="With --worker: stop after this many jobs (useful for testing).",
    )

    parser.add_argument( "--dry-run", action="store_true",
        help=( "With --worker: embed the first pending job but write nothing and leave jobs untouched." ),
    )

    args = parser.parse_args()

    if args.repair is not None and (
        args.corpus is not None
        or args.doc_id is not None
    ):
        parser.error( "--repair cannot be combined with --corpus or --doc-id" )

    if args.repair is not None and args.repair_years:
        parser.error( "--repair and --repair-years cannot be used together" )

    if args.index_only and any(
        (
            args.repair,
            args.repair_years,
            args.add_scale,
            args.phrases,
        )
    ):
        parser.error( "--index-only cannot be combined with processing, repair, or backfill options" )

    # Job queue validation:

    if args.populate and args.worker:
        parser.error( "--populate and --worker cannot be used together" )

    if (args.populate or args.worker) and (
        args.repair is not None
        or args.repair_years
        or args.index_only
    ):
        parser.error(
            "--populate/--worker cannot be combined with "
            "--repair, --repair-years or --index-only"
        )

    if args.worker and (
        args.corpus is not None
        or args.doc_id is not None
    ):
        parser.error(
            "--worker claims whatever is queued; narrow the queue with "
            "--corpus/--doc-id at --populate time instead"
        )

    # These only act on the queue. Rejecting them elsewhere means a
    # direct run can never be mistaken for a dry run.
    if args.dry_run and not args.worker:
        parser.error("--dry-run requires --worker")

    if (
        args.worker_id is not None
        or args.max_docs is not None
    ) and not args.worker:
        parser.error("--worker-id and --max-docs require --worker")

    if (
        args.min_year is not None
        or args.max_year is not None
    ) and not args.populate:
        parser.error("--min-year and --max-year require --populate")

    return args


def main() -> None:
    args = parse_args()

    torch.set_num_threads( int( os.environ.get( "OMP_NUM_THREADS", "4", ) ) )
    torch.set_num_interop_threads(1)

    conn = get_connection( application_name="tier1-creater", )

    if not args.worker and not os.environ.get("COLAB_MODE"):
        create_events_table(conn)

    if args.index_only:
        try:
            writer = EventWriter( conn, args.lance_root, )
            writer.index_existing_tables()
        finally:
            conn.close()

        return

    if args.populate:
        # Enqueue only: needs the database, not MacBERTh or Lance.
        try:
            ensure_jobs_table(conn)
            kind = job_kind( phrases=args.phrases, add_scale=args.add_scale, )
            logger.info( "[tier1] populating %s jobs", kind, )

            processor = CorpusProcessor(
                conn,
                None,
                None,
                neighbour_radius=args.neighbour_radius,
                report_every=args.report_every,
            )

            created = processor.populate_jobs(
                kind,
                corpus=args.corpus,
                doc_id=args.doc_id,
                min_year=args.min_year,
                max_year=args.max_year,
            )
            logger.info( "[tier1] populated %d %s jobs", created, kind, )

        finally:
            conn.close()

        return

    try:
        if not args.worker:
            ensure_jobs_table(conn)

        mac = load_macberth()

        pipeline = MacBERThPipeline(
            mac,
            batch_size=args.batch_size,
            mask_targets=args.mask,
        )

        writer = EventWriter( conn, args.lance_root )

        processor = CorpusProcessor(
            conn,
            pipeline,
            writer,
            neighbour_radius=args.neighbour_radius,
            report_every=args.report_every,
        )

        if args.worker:
            kind = job_kind( phrases=args.phrases, add_scale=args.add_scale, )
            processor.process_queue(
                kind=kind,
                worker_id=args.worker_id,
                max_docs=args.max_docs,
                dry_run=args.dry_run,
                skip_indexing=args.skip_indexing,
            )

            if backfill_scale_for_kind(kind) is not None:
                # Like --add-scale: indexing of the backfilled scalewas already handled by process_queue().
                return

        elif args.repair:
            corpus, doc_id = args.repair
            processor.repair( corpus=corpus, doc_id=doc_id, phrases=args.phrases, )

        elif args.repair_years:
            start_year, end_year = args.repair_years
            repair_year_range(
                processor=processor,
                conn=conn,
                start_year=start_year,
                end_year=end_year,
                corpus=args.corpus,
            )

        elif args.add_scale:
            processor.backfill_scale( args.add_scale, corpus=args.corpus, doc_id=args.doc_id )
            return

        elif args.phrases:
            processor.process_phrases( corpus=args.corpus, doc_id=args.doc_id, )

        else:
            processor.process( corpus=args.corpus, doc_id=args.doc_id, )

        if args.dry_run:
            logger.info( "[tier1] --dry-run set; not building indexes" )
        elif args.skip_indexing:
            logger.info( "[tier1] --skip-indexing set; leaving index (re)build for a later run" )
        else:
            writer.build_indexes()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
