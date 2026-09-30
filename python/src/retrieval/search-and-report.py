#!/usr/bin/env python
"""
Probe MacBERTh phrase semantics against the existing chronological Lance
observation indexes, display PostgreSQL token context around each hit, and
compare retrieval overlap across independent semantic probes.

A phrase is encoded independently of the corpus and used directly as a
query vector. Lance returns the nearest corpus observations; PostgreSQL
then supplies authoritative event provenance and source-token context.

No PostgreSQL data is modified.

The comparison stage retains each query's independent rank and distance.
It identifies events and documents retrieved by multiple distinct probes
without collapsing their distances into a single semantic score.

The complete result is written as a self-contained dark-theme HTML report.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

from lib.corpus_config import LANCE_INDEXES_DIR, OUT_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger

from retrieval.macberth_phrase_encoder2 import (
    MacBertMeanPhraseEncoder,
)
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace


SCALE = "medium"
MIN_YEAR = 1500
MAX_YEAR = 1949
TOP_N = 20
CONTEXT_TOKENS = 30

CARRIER = "The person had {}."

PHRASES = (
    "white hair",
    "white beard",
    "white skin",
    "white face",
    "white eyes",
    "pale eyes",
    "bright eyes",
    "clear eyes",
    "light eyes",
    "grey eyes",
    "gray eyes",
    "blue eyes",
    "pale blue eyes",
    "shining eyes",
    "luminous eyes",
    "glowing eyes",
    "strange eyes",
    "piercing eyes",
)

MIN_EVENT_QUERY_OVERLAP = 2
MIN_DOCUMENT_QUERY_OVERLAP = 2

REPORT_PATH = OUT_DIR / "phrase_probe_report.html"


def fetch_event_metadata(
    connection,
    event_ids: list[int],
) -> dict[int, dict]:
    """
    Fetch authoritative event provenance and source-document metadata.

    Events remain the authoritative source for token-level provenance;
    documents supplies human-readable bibliographic metadata.
    """

    if not event_ids:
        return {}

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                e.event_id,
                e.corpus,
                e.doc_id,
                e.token_idx,
                e.token,
                e.pub_year,
                d.title,
                d.author
            FROM events AS e
            LEFT JOIN documents AS d
              ON d.corpus = e.corpus
             AND d.doc_id = e.doc_id
            WHERE e.event_id = ANY(%s)
            """,
            (event_ids,),
        )

        rows = cursor.fetchall()

    metadata = {}

    for row in rows:

        (
            event_id,
            corpus,
            doc_id,
            token_idx,
            token,
            pub_year,
            title,
            author,
        ) = row

        metadata[int(event_id)] = {
            "event_id": int(event_id),
            "corpus": str(corpus),
            "doc_id": str(doc_id),
            "token_idx": int(token_idx),
            "token": str(token),
            "pub_year": (
                int(pub_year)
                if pub_year is not None
                else None
            ),
            "title": (
                str(title)
                if title is not None
                else None
            ),
            "author": (
                str(author)
                if author is not None
                else None
            ),
        }

    missing = set(event_ids) - set(metadata)

    if missing:
        raise RuntimeError(
            "Lance returned event IDs absent from PostgreSQL: "
            f"{sorted(missing)[:10]}"
        )

    return metadata


def fetch_context(
    connection,
    *,
    corpus: str,
    doc_id: str,
    token_idx: int,
    radius: int,
) -> list[tuple[int, str]]:
    """
    Fetch the source token sequence around one event.

    Token position is the authoritative document-local coordinate, so the
    context remains correct even when event IDs are sparse or selective.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                token_idx,
                token
            FROM tokens
            WHERE corpus = %s
              AND doc_id = %s
              AND token_idx BETWEEN %s AND %s
            ORDER BY token_idx
            """,
            (
                corpus,
                doc_id,
                max(0, token_idx - radius),
                token_idx + radius,
            ),
        )

        return [
            (
                int(row[0]),
                str(row[1]),
            )
            for row in cursor.fetchall()
        ]


def query_terms(
    query: str,
) -> set[str]:
    """Return literal token terms represented by a probe phrase."""

    return {
        term.lower()
        for term in query.split()
        if term.strip()
    }


def format_context(
    tokens: list[tuple[int, str]],
    event_token_idx: int,
    queries: str | list[str] | tuple[str, ...],
) -> str:
    """
    Highlight the retrieved event and literal terms from one or more
    semantic probes.

    The retrieved event is identified by token_idx rather than token text,
    so another occurrence of the same word elsewhere in the context is not
    incorrectly treated as the retrieval result.

    Query highlighting is literal only. A semantic neighbour does not need
    to contain any query term.
    """

    if isinstance(queries, str):
        queries = (queries,)

    literal_terms: set[str] = set()

    for query in queries:
        literal_terms.update(
            query_terms(query)
        )

    parts: list[str] = []

    for token_idx, token in tokens:

        escaped = escape(token)
        token_lower = token.lower()

        is_event = token_idx == event_token_idx
        is_query_term = token_lower in literal_terms

        if is_event and is_query_term:

            parts.append(
                '<span class="retrieved query-match">'
                f"{escaped}"
                "</span>"
            )

        elif is_event:

            parts.append(
                '<span class="retrieved">'
                f"{escaped}"
                "</span>"
            )

        elif is_query_term:

            parts.append(
                '<span class="query-match">'
                f"{escaped}"
                "</span>"
            )

        else:

            parts.append(escaped)

    return " ".join(parts)


def record_result(
    comparison: dict,
    *,
    bucket: tuple[int, int],
    phrase: str,
    rank: int,
    distance: float,
    event: dict,
    context_tokens: list[tuple[int, str]],
    context: str,
) -> None:
    """
    Record one retrieval while retaining query-specific rank, distance,
    provenance, and raw context.

    Raw context tokens are retained separately from rendered HTML so that
    the blended view can apply literal highlighting for every probe that
    retrieved the same event.
    """

    bucket_start, bucket_end = bucket

    event_key = (
        bucket_start,
        bucket_end,
        event["event_id"],
    )

    record = comparison.setdefault(
        event_key,
        {
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "event_id": event["event_id"],
            "corpus": event["corpus"],
            "doc_id": event["doc_id"],
            "token_idx": event["token_idx"],
            "token": event["token"],
            "pub_year": event["pub_year"],
            "title": event["title"],
            "author": event["author"],
            "queries": {},
            "contexts": {},
            "context_tokens": {},
        },
    )

    record["queries"][phrase] = {
        "rank": rank,
        "distance": distance,
    }

    record["contexts"][phrase] = context

    record["context_tokens"][phrase] = context_tokens


def search_phrase(
    *,
    phrase: str,
    encoder: MacBertMeanPhraseEncoder,
    store: LanceObservationIndexStore,
    connection,
    comparison: dict,
    raw_results: dict,
) -> None:
    """
    Search one MacBERTh phrase independently in each physical Lance
    bucket and retain source context for every returned event.

    Results are deliberately not fused with RRF. The purpose is to inspect
    the raw semantic neighbourhood within each temporal population.
    """

    query_vector = encoder.encode(
        phrase,
        CARRIER,
    )

    if query_vector.shape != (768,):
        raise RuntimeError(
            "Unexpected MacBERTh query-vector shape: "
            f"{query_vector.shape}"
        )

    logger.info(
        "[phrase-probe] phrase=%r carrier=%r vector_shape=%s",
        phrase,
        CARRIER,
        query_vector.shape,
    )

    queries_by_scale = {
        SCALE: query_vector,
    }

    search_space = SearchSpace(
        years=(MIN_YEAR, MAX_YEAR),
        scale=(SCALE,),
    )

    phrase_results = raw_results.setdefault(
        phrase,
        {},
    )

    for (
        bucket_start,
        bucket_end,
    ), results_by_scale in store.diachronic_search(
        queries_by_scale,
        search_space,
        k=TOP_N,
    ):

        result = results_by_scale[SCALE]

        event_ids = result.event_ids
        distances = result.distances

        valid = [
            (
                int(event_id),
                float(distance),
            )
            for event_id, distance in zip(
                event_ids,
                distances,
            )
            if int(event_id) >= 0
        ]

        if not valid:
            continue

        metadata = fetch_event_metadata(
            connection,
            [
                event_id
                for event_id, _ in valid
            ],
        )

        bucket_results = []

        for rank, (
            event_id,
            distance,
        ) in enumerate(
            valid,
            start=1,
        ):

            event = metadata[event_id]

            context = fetch_context(
                connection,
                corpus=event["corpus"],
                doc_id=event["doc_id"],
                token_idx=event["token_idx"],
                radius=CONTEXT_TOKENS,
            )

            context_html = format_context(
                context,
                event["token_idx"],
                phrase,
            )

            record_result(
                comparison,
                bucket=(
                    bucket_start,
                    bucket_end,
                ),
                phrase=phrase,
                rank=rank,
                distance=distance,
                event=event,
                context_tokens=context,
                context=context_html,
            )

            bucket_results.append(
                {
                    "rank": rank,
                    "distance": distance,
                    "event": event,
                    "context": context_html,
                }
            )

        phrase_results[
            (bucket_start, bucket_end)
        ] = bucket_results


def html_escape(value) -> str:
    """Convert nullable metadata to safe HTML text."""

    if value is None:
        return "—"

    return escape(str(value))


def render_raw_results(
    raw_results: dict,
) -> str:
    """Render complete per-query retrieval results as collapsible sections."""

    sections = []

    for phrase in PHRASES:

        buckets = raw_results.get(
            phrase,
            {},
        )

        bucket_sections = []

        for (
            bucket_start,
            bucket_end,
        ), results in buckets.items():

            rows = []

            for result in results:

                event = result["event"]

                rows.append(
                    f"""
                    <article class="result">
                        <div class="result-header">
                            <span class="rank">#{result["rank"]}</span>
                            <span class="distance">
                                distance={result["distance"]:.6f}
                            </span>
                            <span class="year">
                                {html_escape(event["pub_year"])}
                            </span>
                            <span class="event">
                                event={event["event_id"]}
                            </span>
                        </div>

                        <div class="metadata">
                            <div>
                                <strong>Author</strong>
                                {html_escape(event["author"])}
                            </div>
                            <div>
                                <strong>Title</strong>
                                {html_escape(event["title"])}
                            </div>
                            <div>
                                <strong>Corpus / document</strong>
                                {html_escape(event["corpus"])}
                                /
                                {html_escape(event["doc_id"])}
                            </div>
                            <div>
                                <strong>Retrieved token</strong>
                                {html_escape(event["token"])}
                                &nbsp;
                                <strong>index</strong>
                                {event["token_idx"]}
                            </div>
                        </div>

                        <div class="context">
                            {result["context"]}
                        </div>
                    </article>
                    """
                )

            bucket_sections.append(
                f"""
                <details class="bucket">
                    <summary>
                        {bucket_start}–{bucket_end}
                        <span class="count">
                            {len(results)} results
                        </span>
                    </summary>
                    {"".join(rows)}
                </details>
                """
            )

        sections.append(
            f"""
            <details class="query">
                <summary>
                    <span class="query-name">{escape(phrase)}</span>
                    <span class="count">
                        {sum(len(x) for x in buckets.values())} results
                    </span>
                </summary>

                {"".join(bucket_sections)}
            </details>
            """
        )

    return "".join(sections)


def render_event_comparison(
    comparison: dict,
) -> str:
    """
    Render events retrieved by multiple independent semantic probes.

    Query count measures convergence across probe phrases rather than
    treating repeated retrieval of the same event as independent evidence.
    """

    records = [
        record
        for record in comparison.values()
        if len(record["queries"]) >= MIN_EVENT_QUERY_OVERLAP
    ]

    records.sort(
        key=lambda record: (
            record["bucket_start"],
            -len(record["queries"]),
            record["pub_year"] or 0,
            record["event_id"],
        ),
    )

    if not records:
        return (
            '<p class="empty">'
            f"No events were retrieved by "
            f"{MIN_EVENT_QUERY_OVERLAP} or more distinct queries."
            "</p>"
        )

    sections = []
    current_bucket = None

    for record in records:

        bucket = (
            record["bucket_start"],
            record["bucket_end"],
        )

        if bucket != current_bucket:

            sections.append(
                f"""
                <h3 class="bucket-heading">
                    {bucket[0]}–{bucket[1]}
                </h3>
                """
            )

            current_bucket = bucket

        query_rows = []

        for phrase, result in sorted(
            record["queries"].items(),
            key=lambda item: (
                item[1]["distance"],
                item[0],
            ),
        ):

            query_rows.append(
                f"""
                <tr>
                    <td>{escape(phrase)}</td>
                    <td>{result["rank"]}</td>
                    <td>{result["distance"]:.6f}</td>
                </tr>
                """
            )

        context = format_context(
            next(
                iter(record["context_tokens"].values()),
                [],
            ),
            record["token_idx"],
            tuple(record["queries"]),
        )

        sections.append(
            f"""
            <article class="comparison-card">
                <div class="comparison-header">
                    <span class="overlap">
                        {len(record["queries"])} queries
                    </span>
                    <span>
                        {html_escape(record["pub_year"])}
                    </span>
                    <span>
                        event {record["event_id"]}
                    </span>
                    <span>
                        {html_escape(record["corpus"])}
                        /
                        {html_escape(record["doc_id"])}
                    </span>
                </div>

                <div class="metadata">
                    <div>
                        <strong>Author</strong>
                        {html_escape(record["author"])}
                    </div>
                    <div>
                        <strong>Title</strong>
                        {html_escape(record["title"])}
                    </div>
                    <div>
                        <strong>Retrieved token</strong>
                        {html_escape(record["token"])}
                        &nbsp;
                        <strong>index</strong>
                        {record["token_idx"]}
                    </div>
                </div>

                <div class="context">
                    {context}
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>Query</th>
                            <th>Rank</th>
                            <th>Distance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(query_rows)}
                    </tbody>
                </table>
            </article>
            """
        )

    return "".join(sections)


def render_document_comparison(
    comparison: dict,
) -> str:
    """
    Render source documents containing events retrieved by multiple probes.

    Document-level convergence is kept separate from event-level
    convergence because several retrieved events can belong to one source.
    """

    documents: dict[tuple[int, int, str, str], dict] = {}

    for record in comparison.values():

        key = (
            record["bucket_start"],
            record["bucket_end"],
            record["corpus"],
            record["doc_id"],
        )

        document = documents.setdefault(
            key,
            {
                "bucket_start": record["bucket_start"],
                "bucket_end": record["bucket_end"],
                "corpus": record["corpus"],
                "doc_id": record["doc_id"],
                "pub_year": record["pub_year"],
                "title": record["title"],
                "author": record["author"],
                "events": {},
                "queries": defaultdict(list),
            },
        )

        document["events"][record["event_id"]] = record

        for phrase, result in record["queries"].items():

            document["queries"][phrase].append(
                {
                    "event_id": record["event_id"],
                    "rank": result["rank"],
                    "distance": result["distance"],
                }
            )

    records = [
        document
        for document in documents.values()
        if len(document["queries"]) >= MIN_DOCUMENT_QUERY_OVERLAP
    ]

    records.sort(
        key=lambda record: (
            record["bucket_start"],
            -len(record["queries"]),
            record["pub_year"] or 0,
            record["doc_id"],
        ),
    )

    if not records:
        return (
            '<p class="empty">'
            f"No documents were retrieved by "
            f"{MIN_DOCUMENT_QUERY_OVERLAP} or more distinct queries."
            "</p>"
        )

    sections = []
    current_bucket = None

    for record in records:

        bucket = (
            record["bucket_start"],
            record["bucket_end"],
        )

        if bucket != current_bucket:

            sections.append(
                f"""
                <h3 class="bucket-heading">
                    {bucket[0]}–{bucket[1]}
                </h3>
                """
            )

            current_bucket = bucket

        query_rows = []

        for phrase, results in sorted(
            record["queries"].items(),
        ):

            best = min(
                results,
                key=lambda result: result["distance"],
            )

            query_rows.append(
                f"""
                <tr>
                    <td>{escape(phrase)}</td>
                    <td>{len(results)}</td>
                    <td>{best["rank"]}</td>
                    <td>{best["distance"]:.6f}</td>
                </tr>
                """
            )

        sections.append(
            f"""
            <article class="comparison-card">
                <div class="comparison-header">
                    <span class="overlap">
                        {len(record["queries"])} queries
                    </span>
                    <span>
                        {len(record["events"])} events
                    </span>
                    <span>
                        {html_escape(record["pub_year"])}
                    </span>
                    <span>
                        {html_escape(record["corpus"])}
                        /
                        {html_escape(record["doc_id"])}
                    </span>
                </div>

                <div class="metadata">
                    <div>
                        <strong>Author</strong>
                        {html_escape(record["author"])}
                    </div>
                    <div>
                        <strong>Title</strong>
                        {html_escape(record["title"])}
                    </div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>Query</th>
                            <th>Events</th>
                            <th>Best rank</th>
                            <th>Best distance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(query_rows)}
                    </tbody>
                </table>
            </article>
            """
        )

    return "".join(sections)


def render_blended_quotes(
    comparison: dict,
) -> str:
    """
    Render all semantically retrieved passages as a chronological stream.

    Each year is collapsible so the report remains navigable even when many
    observations are retrieved. An event appears once within the stream,
    while every probe that retrieved it remains visible as independent
    evidence.

    Literal query highlighting is recomputed from the raw context using all
    probes that retrieved the event, rather than using the first rendered
    context only.
    """

    records = list(comparison.values())

    records.sort(
        key=lambda record: (
            record["pub_year"]
            if record["pub_year"] is not None
            else 9999,
            record["bucket_start"],
            record["event_id"],
        ),
    )

    if not records:
        return (
            '<p class="empty">'
            "No retrieved events."
            "</p>"
        )

    grouped: dict[int | None, list[dict]] = defaultdict(list)

    for record in records:
        grouped[
            record["pub_year"]
        ].append(record)

    sections: list[str] = []

    for year, year_records in grouped.items():

        if year is None:
            heading = "Unknown year"
        else:
            heading = str(year)

        cards: list[str] = []

        for record in year_records:

            query_rows = []

            for phrase, result in sorted(
                record["queries"].items(),
                key=lambda item: (
                    item[1]["distance"],
                    item[0],
                ),
            ):

                query_rows.append(
                    f"""
                    <span class="probe">
                        {escape(phrase)}
                        <span class="probe-detail">
                            rank {result["rank"]},
                            distance {result["distance"]:.6f}
                        </span>
                    </span>
                    """
                )

            context_tokens = next(
                iter(record["context_tokens"].values()),
                [],
            )

            context = format_context(
                context_tokens,
                record["token_idx"],
                tuple(record["queries"]),
            )

            cards.append(
                f"""
                <article class="quote-card">

                    <div class="quote-header">
                        <span class="year">
                            {html_escape(record["pub_year"])}
                        </span>

                        <span class="event">
                            event {record["event_id"]}
                        </span>

                        <span class="event">
                            {html_escape(record["corpus"])}
                            /
                            {html_escape(record["doc_id"])}
                        </span>
                    </div>

                    <div class="metadata">
                        <div>
                            <strong>Author</strong>
                            {html_escape(record["author"])}
                        </div>

                        <div>
                            <strong>Title</strong>
                            {html_escape(record["title"])}
                        </div>

                        <div>
                            <strong>Retrieved token</strong>
                            {html_escape(record["token"])}
                            &nbsp;
                            <strong>index</strong>
                            {record["token_idx"]}
                        </div>

                        <div>
                            <strong>Bucket</strong>
                            {record["bucket_start"]}–{record["bucket_end"]}
                        </div>
                    </div>

                    <div class="context">
                        {context}
                    </div>

                    <div class="probes">
                        {"".join(query_rows)}
                    </div>

                </article>
                """
            )

        sections.append(
            f"""
            <details class="year-section">
                <summary>
                    <span class="year-label">
                        {heading}
                    </span>
                    <span class="count">
                        {len(year_records)} retrieved events
                    </span>
                </summary>

                {"".join(cards)}
            </details>
            """
        )

    return "".join(sections)


def render_toc() -> str:
    """Render stable navigation links to the major report sections."""

    return """
    <nav class="toc" aria-label="Table of contents">

        <div class="toc-title">
            Contents
        </div>

        <a href="#overview">
            Overview
        </a>

        <a href="#event-retrieval">
            Cross-query event retrieval
        </a>

        <a href="#document-retrieval">
            Cross-query document retrieval
        </a>

        <a href="#blended-quotes">
            Blended quotes by year
        </a>

        <a href="#raw-results">
            Raw retrieval results
        </a>

    </nav>
    """


def render_report(
    *,
    comparison: dict,
    raw_results: dict,
) -> str:
    """Build the complete self-contained HTML report."""

    event_overlap_count = sum(
        1
        for record in comparison.values()
        if len(record["queries"]) >= MIN_EVENT_QUERY_OVERLAP
    )

    documents = {
        (
            record["bucket_start"],
            record["bucket_end"],
            record["corpus"],
            record["doc_id"],
        )
        for record in comparison.values()
        if len(record["queries"]) >= MIN_DOCUMENT_QUERY_OVERLAP
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MacBERTh Phrase Probe</title>

<style>
:root {{
    color-scheme: dark;
    --bg: #111315;
    --panel: #181b1f;
    --panel-alt: #20242a;
    --border: #343a42;
    --text: #e7e9ec;
    --muted: #9da5ae;
    --accent: #7db7ff;
    --accent-soft: #263b55;
    --query-match: rgba(100, 180, 255, 0.28);
    --hit: rgba(255, 193, 7, 0.5);
}}

* {{
    box-sizing: border-box;
}}

html {{
    background: var(--bg);
    scroll-behavior: smooth;
}}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    font-size: 17px;
    line-height: 1.55;
}}

main {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 32px;
}}

h1 {{
    margin: 0 0 8px;
    font-size: 2rem;
}}

h2 {{
    margin-top: 48px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    scroll-margin-top: 24px;
}}

h3 {{
    margin-top: 28px;
}}

.subtitle {{
    color: var(--muted);
    margin-bottom: 28px;
}}

.toc {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin: 28px 0;
    padding: 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
}}

.toc-title {{
    color: var(--muted);
    font-weight: 700;
    margin-right: 8px;
}}

.toc a {{
    color: var(--accent);
    text-decoration: none;
    padding: 5px 9px;
    border-radius: 5px;
    background: var(--accent-soft);
}}

.toc a:hover {{
    text-decoration: underline;
}}

.config {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 24px 0;
}}

.stat {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
}}

.stat-label {{
    color: var(--muted);
    font-size: 0.85rem;
}}

.stat-value {{
    margin-top: 3px;
    font-weight: 600;
}}

details {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 10px 0;
}}

summary {{
    cursor: pointer;
    padding: 12px 16px;
    font-weight: 600;
    list-style-position: outside;
}}

summary:hover {{
    background: var(--panel-alt);
}}

.query > summary {{
    font-size: 1.05rem;
}}

.bucket {{
    margin: 8px;
    background: var(--panel-alt);
}}

.query-name {{
    color: var(--accent);
}}

.count {{
    float: right;
    color: var(--muted);
    font-weight: normal;
}}

.year-section {{
    margin: 8px 0;
    background: var(--panel-alt);
    scroll-margin-top: 24px;
}}

.year-section > summary {{
    font-size: 1.05rem;
}}

.year-label {{
    color: var(--accent);
    font-weight: 700;
}}

.result {{
    margin: 10px;
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--panel);
}}

.result-header,
.comparison-header,
.quote-header {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: baseline;
    margin-bottom: 10px;
}}

.rank {{
    color: var(--accent);
    font-weight: 700;
}}

.distance,
.year,
.event {{
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.9rem;
}}

.metadata {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 6px 24px;
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 14px;
}}

.metadata strong {{
    color: var(--text);
}}

.context {{
    padding: 14px 16px;
    background: #0c0e10;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family:
        ui-monospace,
        SFMono-Regular,
        Consolas,
        "Liberation Mono",
        monospace;
    font-size: 0.95rem;
    line-height: 1.8;
    overflow-wrap: anywhere;
}}

.retrieved {{
    background: var(--hit);
    border-radius: 3px;
    padding: 2px 4px;
}}

.query-match {{
    background: var(--query-match);
    border-radius: 3px;
    padding: 2px 4px;
}}

.retrieved.query-match {{
    background:
        linear-gradient(
            var(--hit),
            var(--hit)
        ),
        linear-gradient(
            var(--query-match),
            var(--query-match)
        );
}}

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin: 18px 0;
    color: var(--muted);
    font-size: 0.9rem;
}}

.legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
}}

.legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    display: inline-block;
}}

.legend-swatch.retrieved {{
    background: var(--hit);
}}

.legend-swatch.query {{
    background: var(--query-match);
}}

.bucket-heading {{
    color: var(--accent);
    margin-top: 32px;
}}

.comparison-card,
.quote-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
    margin: 12px;
}}

.overlap {{
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 5px;
    padding: 3px 8px;
    font-weight: 700;
}}

.probes {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}}

.probe {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 5px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 0.88rem;
}}

.probe-detail {{
    color: var(--muted);
    margin-left: 6px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
    font-size: 0.92rem;
}}

th,
td {{
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
}}

th {{
    color: var(--muted);
    font-weight: 600;
}}

.empty {{
    color: var(--muted);
    padding: 20px;
    background: var(--panel);
    border-radius: 8px;
}}

footer {{
    margin-top: 60px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.85rem;
}}

code {{
    color: var(--accent);
}}

@media (max-width: 700px) {{
    main {{
        padding: 16px;
    }}

    body {{
        font-size: 16px;
    }}

    .toc {{
        align-items: stretch;
        flex-direction: column;
    }}

    .toc-title {{
        margin-right: 0;
    }}
}}
</style>
</head>

<body>
<main>

<h1>MacBERTh phrase probe</h1>

<div class="subtitle">
    Semantic retrieval across chronological Lance observation indexes,
    with PostgreSQL provenance and cross-query convergence.
</div>

{render_toc()}

<section id="overview">

<h2>Overview</h2>

<div class="config">
    <div class="stat">
        <div class="stat-label">Corpus years</div>
        <div class="stat-value">{MIN_YEAR}–{MAX_YEAR}</div>
    </div>

    <div class="stat">
        <div class="stat-label">Scale</div>
        <div class="stat-value">{escape(SCALE)}</div>
    </div>

    <div class="stat">
        <div class="stat-label">Results per bucket</div>
        <div class="stat-value">{TOP_N}</div>
    </div>

    <div class="stat">
        <div class="stat-label">Context radius</div>
        <div class="stat-value">{CONTEXT_TOKENS} tokens</div>
    </div>

    <div class="stat">
        <div class="stat-label">Queries</div>
        <div class="stat-value">{len(PHRASES)}</div>
    </div>

    <div class="stat">
        <div class="stat-label">Event overlaps</div>
        <div class="stat-value">{event_overlap_count}</div>
    </div>

    <div class="stat">
        <div class="stat-label">Document overlaps</div>
        <div class="stat-value">{len(documents)}</div>
    </div>
</div>

<p>
    Carrier:
    <code>{escape(CARRIER)}</code>
</p>

<div class="legend">
    <span class="legend-item">
        <span class="legend-swatch retrieved"></span>
        Retrieved corpus token
    </span>

    <span class="legend-item">
        <span class="legend-swatch query"></span>
        Literal query term
    </span>
</div>

<p class="subtitle">
    The retrieved token is the actual corpus observation returned by
    LanceDB. A semantic result does not need to contain any literal query
    term. Literal query highlighting is therefore independent of semantic
    retrieval.
</p>

</section>

<section id="event-retrieval">

<h2>Cross-query event retrieval</h2>

<details class="major-section">
    <summary>
        Events retrieved by multiple independent semantic probes
        <span class="count">{event_overlap_count} events</span>
    </summary>

    <p class="subtitle">
        Events independently retrieved by at least
        {MIN_EVENT_QUERY_OVERLAP} distinct semantic probes.
        Distances and ranks remain query-specific.
    </p>

    {render_event_comparison(comparison)}

</details>

</section>

<section id="document-retrieval">

<h2>Cross-query document retrieval</h2>

<details class="major-section">
    <summary>
        Source documents retrieved by multiple probes
        <span class="count">{len(documents)} documents</span>
    </summary>

    <p class="subtitle">
        Source documents containing events retrieved by at least
        {MIN_DOCUMENT_QUERY_OVERLAP} distinct semantic probes.
        Multiple events from the same document are counted separately from
        query convergence.
    </p>

    {render_document_comparison(comparison)}

</details>

</section>

<section id="blended-quotes">

<h2>Blended quotes by year</h2>

<p class="subtitle">
    All semantically retrieved events from all probe phrases, ordered
    chronologically. Expand a year to inspect its retrieved passages.
    Events retrieved by multiple probes are shown once, with all of their
    independent query ranks and distances retained.
</p>

{render_blended_quotes(comparison)}

</section>

<section id="raw-results">

<h2>Raw retrieval results</h2>

<details class="major-section">

    <summary>
        Complete per-query retrieval results
        <span class="count">{len(PHRASES)} queries</span>
    </summary>

    <p class="subtitle">
        Results grouped by phrase and chronological bucket. Expand a query
        and bucket to inspect the individual passages.
    </p>

    {render_raw_results(raw_results)}

</details>

</section>

<footer>
    Generated by the MacBERTh semantic phrase probe.
    No PostgreSQL data was modified.
</footer>

</main>
</body>
</html>
"""


def main() -> None:

    logger.info(
        "[phrase-probe] Lance root=%s",
        LANCE_INDEXES_DIR,
    )

    store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR,
        available_years=range(
            MIN_YEAR,
            MAX_YEAR + 1,
        ),
        available_scales=(SCALE,),
    )

    encoder = MacBertMeanPhraseEncoder()

    comparison = {}
    raw_results = {}

    with get_connection(
        application_name="semantic-phrase-probe",
    ) as connection:

        for phrase in PHRASES:

            search_phrase(
                phrase=phrase,
                encoder=encoder,
                store=store,
                connection=connection,
                comparison=comparison,
                raw_results=raw_results,
            )

    report = render_report(
        comparison=comparison,
        raw_results=raw_results,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    event_overlap_count = sum(
        1
        for record in comparison.values()
        if len(record["queries"]) >= MIN_EVENT_QUERY_OVERLAP
    )

    document_keys = {
        (
            record["bucket_start"],
            record["bucket_end"],
            record["corpus"],
            record["doc_id"],
        )
        for record in comparison.values()
        if len(record["queries"]) >= MIN_DOCUMENT_QUERY_OVERLAP
    }

    logger.info(
        "[phrase-probe] event overlaps=%d",
        event_overlap_count,
    )

    logger.info(
        "[phrase-probe] document overlaps=%d",
        len(document_keys),
    )

    logger.info(
        "[phrase-probe] HTML report=%s",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
