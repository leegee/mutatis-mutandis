#!/usr/bin/env python
"""
test_phrase_search5.py

Generate a candidate-selection report from the existing MacBERTh phrase
probe.

The semantic retrieval stage remains unchanged:

    probe phrase
        |
        v
    MacBERTh vector
        |
        v
    LanceDB chronological search
        |
        v
    PostgreSQL event/document provenance

This report adds a second-stage reduction layer intended to make the
results human-readable.

Semantic retrieval is deliberately restricted to the two hair-related
probes:

    white hair
    white beard

After retrieval, candidate passages are cross-referenced for historical
eye vocabulary:

    eye
    eyes
    eyne
    eie
    eien

Eye vocabulary is therefore an evidential annotation, not a semantic
query. It does not affect MacBERTh vectors, LanceDB distances, or
candidate ranking.

Nearby retrieved events in the same source document are grouped into
candidate passages. Candidates are then ordered by transparent
evidential priority:

    1. number of distinct probe families represented
    2. weighted distinct-probe evidence
    3. number of distinct probe phrases represented
    4. number of retrieved events represented
    5. best semantic distance

The complete raw retrieval data remains available in the report.

The candidate selector is not a historical classifier. It does not decide
whether a passage is divine, pathological, racialised, supernatural, etc.
Its purpose is only to reduce the amount of material requiring human
inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

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
MIN_YEAR = 1900
MAX_YEAR = 2000

TOP_N = 20
CONTEXT_TOKENS = 30

CARRIER = "The person had {}."

# Only hair-related probes drive semantic retrieval. Eye vocabulary is
# deliberately checked only after retrieval so that it cannot distort the
# semantic search space.
PHRASES = (
    "white hair",
    # "white beard",
)

PROBE_FAMILIES = {
    "hair": {
        "white hair",
        # "white beard",
    },
}

# Probe weights affect only candidate reading priority. They must not be
# applied to MacBERTh vectors or LanceDB distances: those remain the
# empirical retrieval signal.
PROBE_WEIGHTS = {
    "white hair": 2.0,
    # "white beard": 0.5,
}

# These are historical lexical cross-references applied after semantic
# retrieval. They are not semantic probes.
EYE_TERMS = {
    "eye",
    "eyes",
    "eyne",
    "eie",
    "eien",
}

# Retained for reporting bodily vocabulary in retrieved passages. This is
# descriptive evidence only; it is not a candidate-selection gate.
BODILY_TERMS = {
    "eye",
    "eyes",
    "eyne",
    "eie",
    "eien",
    "hair",
    "haire",
    "heer",
    "beard",
    "berd",
    "skin",
    "skyn",
    "skinn",
    "face",
    "visage",
    "forehead",
    "brow",
    "browe",
    "eyelid",
    "eyelids",
    "lash",
    "lashes",
    "head",
    "hed",
    "cheek",
    "cheeke",
    "flesh",
    "fleshe",
    "complexion",
    "complection",
}

# Retrieved events belonging to the same source document and occurring
# close together are treated as one candidate passage. This prevents a
# single descriptive passage from appearing repeatedly merely because
# several tokens in it were independently retrieved.
PASSAGE_GAP = 60

# The report is deliberately small enough to read manually.
CANDIDATE_LIMIT = 50

MIN_EVENT_QUERY_OVERLAP = 2
MIN_DOCUMENT_QUERY_OVERLAP = 2

# The 1800-1849 Lance bucket is currently known to contain invalid
# retrievals. Keep its raw data available for diagnosis, but exclude it
# from candidate selection and candidate reporting.
INVALID_CANDIDATE_BUCKETS = {
    (1800, 1849),
}

REPORT_PATH = OUT_DIR / "phrase_probe_candidates.html"


@dataclass
class Candidate:
    """A group of nearby retrieved events treated as one reading candidate."""

    corpus: str
    doc_id: str
    pub_year: int | None
    title: str | None
    author: str | None
    bucket_start: int
    bucket_end: int
    events: list[dict]


def query_terms(
    query: str,
) -> set[str]:
    """Return literal token terms represented by a probe phrase."""

    return {
        term.lower()
        for term in query.split()
        if term.strip()
    }


def eye_hits(
    tokens: list[tuple[int, str]],
) -> list[str]:
    """Return distinct eye vocabulary represented in a token sequence."""

    hits: list[str] = []
    seen: set[str] = set()

    for _, token in tokens:

        normalised = token.lower().strip(
            ".,;:!?()[]{}\"'“”‘’"
        )

        if (
            normalised in EYE_TERMS
            and normalised not in seen
        ):

            hits.append(normalised)
            seen.add(normalised)

    return hits


def query_family(
    phrase: str,
) -> str:
    """
    Return the broad probe family for one phrase.

    Every configured phrase must belong to exactly one family. Failing
    loudly here prevents silent changes to the candidate-selection logic.
    """

    matches = [
        family
        for family, phrases in PROBE_FAMILIES.items()
        if phrase in phrases
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Probe phrase has {len(matches)} families: {phrase!r}"
        )

    return matches[0]


def validate_probe_configuration() -> None:
    """
    Verify that the configured probe list, family assignments, and weights
    describe exactly the same set of phrases.

    Configuration drift here would otherwise produce a report whose
    displayed probe count and scoring logic disagree.
    """

    phrase_set = set(PHRASES)

    family_phrases = {
        phrase
        for phrases in PROBE_FAMILIES.values()
        for phrase in phrases
    }

    weight_phrases = set(PROBE_WEIGHTS)

    if phrase_set != family_phrases:
        missing_from_families = phrase_set - family_phrases
        extra_in_families = family_phrases - phrase_set

        raise RuntimeError(
            "Probe/family configuration mismatch: "
            f"missing_from_families={sorted(missing_from_families)}, "
            f"extra_in_families={sorted(extra_in_families)}"
        )

    if phrase_set != weight_phrases:
        missing_weights = phrase_set - weight_phrases
        extra_weights = weight_phrases - phrase_set

        raise RuntimeError(
            "Probe/weight configuration mismatch: "
            f"missing_weights={sorted(missing_weights)}, "
            f"extra_weights={sorted(extra_weights)}"
        )

    for phrase in PHRASES:
        query_family(phrase)

        weight = PROBE_WEIGHTS[phrase]

        if weight <= 0:
            raise RuntimeError(
                f"Probe weight must be positive: {phrase!r}={weight}"
            )


def format_context(
    tokens: list[tuple[int, str]],
    event_token_idx: int,
    queries: tuple[str, ...],
) -> str:
    """
    Highlight the actual retrieved corpus token separately from literal
    terms belonging to any probe that retrieved the event.

    Semantic retrieval and literal highlighting are intentionally separate:
    a semantically retrieved passage need not contain any query term.

    Eye vocabulary is highlighted independently so that it is visible as
    post-retrieval cross-reference evidence.
    """

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
        is_eye_term = (
            token_lower.strip(
                ".,;:!?()[]{}\"'“”‘’"
            )
            in EYE_TERMS
        )

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

        elif is_eye_term:

            parts.append(
                '<span class="eye-match">'
                f"{escaped}"
                "</span>"
            )

        else:

            parts.append(escaped)

    return " ".join(parts)


def fetch_event_metadata(
    connection,
    event_ids: list[int],
) -> dict[int, dict]:
    """Fetch authoritative event and document metadata from PostgreSQL."""

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
    """Fetch source tokens around one authoritative event position."""

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


def add_retrieval(
    comparison: dict,
    *,
    bucket: tuple[int, int],
    phrase: str,
    rank: int,
    distance: float,
    event: dict,
    context_tokens: list[tuple[int, str]],
) -> None:
    """Add one independent semantic retrieval to the comparison index."""

    bucket_start, bucket_end = bucket

    key = (
        bucket_start,
        bucket_end,
        event["event_id"],
    )

    record = comparison.setdefault(
        key,
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
            "context_tokens": context_tokens,
        },
    )

    # The same event always refers to the same source token position.
    # Retaining the first context avoids duplicating identical token data.
    record["queries"][phrase] = {
        "rank": rank,
        "distance": distance,
    }


def search_phrase(
    *,
    phrase: str,
    encoder: MacBertMeanPhraseEncoder,
    store: LanceObservationIndexStore,
    connection,
    comparison: dict,
    raw_results: dict,
) -> None:
    """Run one semantic probe across the configured chronological range."""

    query_vector = encoder.encode(
        phrase,
        CARRIER,
    )

    if query_vector.shape != (768,):
        raise RuntimeError(
            "Unexpected MacBERTh query-vector shape: "
            f"{query_vector.shape}"
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

        valid = [
            (
                int(event_id),
                float(distance),
            )
            for event_id, distance in zip(
                result.event_ids,
                result.distances,
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

            add_retrieval(
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
            )

            bucket_results.append(
                {
                    "rank": rank,
                    "distance": distance,
                    "event": event,
                    "context": format_context(
                        context,
                        event["token_idx"],
                        (phrase,),
                    ),
                }
            )

        phrase_results[
            (bucket_start, bucket_end)
        ] = bucket_results


def bodily_hits(
    candidate_tokens: list[tuple[int, str]],
) -> list[str]:
    """
    Return distinct bodily vocabulary found in a candidate passage.

    This is evidence for display only. It is deliberately not used as a
    candidate-selection gate.
    """

    hits: list[str] = []
    seen: set[str] = set()

    for _, token in candidate_tokens:

        normalised = token.lower().strip(
            ".,;:!?()[]{}\"'“”‘’"
        )

        if (
            normalised in BODILY_TERMS
            and normalised not in seen
        ):

            hits.append(normalised)
            seen.add(normalised)

    return hits


def candidate_bodily_hits(
    candidate: Candidate,
) -> list[str]:
    """Return bodily vocabulary found anywhere in candidate contexts."""

    hits: list[str] = []
    seen: set[str] = set()

    for event in candidate.events:

        for term in bodily_hits(
            event["context_tokens"]
        ):

            if term not in seen:
                hits.append(term)
                seen.add(term)

    return hits


def candidate_eye_hits(
    candidate: Candidate,
) -> list[str]:
    """
    Return eye vocabulary found anywhere in the candidate contexts.

    This is the second-stage cross-reference: eye vocabulary is measured
    after the white-hair semantic search rather than used as a query.
    """

    hits: list[str] = []
    seen: set[str] = set()

    for event in candidate.events:

        for term in eye_hits(
            event["context_tokens"]
        ):

            if term not in seen:
                hits.append(term)
                seen.add(term)

    return hits


def candidate_query_summary(
    candidate: Candidate,
) -> tuple[set[str], set[str]]:
    """Return distinct probe phrases and probe families represented."""

    phrases: set[str] = set()

    for event in candidate.events:
        phrases.update(event["queries"])

    families = {
        query_family(phrase)
        for phrase in phrases
    }

    return phrases, families


def candidate_weighted_score(
    candidate: Candidate,
) -> float:
    """
    Calculate weighted distinct-probe evidence for one candidate.

    Each probe contributes at most once regardless of how many retrieved
    events represent it. This prevents repeated retrieval of one probe from
    overwhelming convergence across different probes.
    """

    phrases, _ = candidate_query_summary(
        candidate
    )

    return sum(
        PROBE_WEIGHTS[phrase]
        for phrase in phrases
    )


def candidate_best_distance(
    candidate: Candidate,
) -> float:
    """Return the strongest LanceDB distance represented by a candidate."""

    distances = [
        result["distance"]
        for event in candidate.events
        for result in event["queries"].values()
    ]

    if not distances:
        raise RuntimeError(
            "Candidate contains no retrieval distances."
        )

    return min(distances)


def build_candidates(
    comparison: dict,
) -> list[Candidate]:
    """
    Group nearby retrieved events into source-level candidate passages.

    Events are grouped only within the same corpus/document and when their
    token positions are sufficiently close. This avoids treating unrelated
    parts of a long document as one candidate while collapsing several
    retrieval hits from the same descriptive passage.

    Known-invalid chronological buckets are retained in raw comparison
    data but excluded here.
    """

    records = [
        record
        for record in comparison.values()
        if (
            record["bucket_start"],
            record["bucket_end"],
        ) not in INVALID_CANDIDATE_BUCKETS
    ]

    records.sort(
        key=lambda record: (
            record["corpus"],
            record["doc_id"],
            record["token_idx"],
            record["event_id"],
        )
    )

    candidates: list[Candidate] = []

    current: Candidate | None = None
    current_last_token_idx: int | None = None

    for record in records:

        starts_new = (
            current is None
            or record["corpus"] != current.corpus
            or record["doc_id"] != current.doc_id
            or current_last_token_idx is None
            or record["token_idx"] - current_last_token_idx
            > PASSAGE_GAP
        )

        if starts_new:

            current = Candidate(
                corpus=record["corpus"],
                doc_id=record["doc_id"],
                pub_year=record["pub_year"],
                title=record["title"],
                author=record["author"],
                bucket_start=record["bucket_start"],
                bucket_end=record["bucket_end"],
                events=[],
            )

            candidates.append(current)

        current.events.append(record)
        current_last_token_idx = record["token_idx"]

    return candidates


def candidate_priority(
    candidate: Candidate,
) -> tuple:
    """
    Return a transparent candidate-reading priority.

    Family count remains the first criterion because convergence across
    different bodily families is more informative than many near-duplicate
    probes from one family.

    Weighted probe evidence is then used to distinguish probes with
    different evidential specificity.

    With the current hair-only probe set, all candidates necessarily belong
    to the same family; this ordering is retained so that adding another
    probe family later does not require changing the selector.
    """

    phrases, families = candidate_query_summary(
        candidate
    )

    weighted_score = candidate_weighted_score(
        candidate
    )

    best_distance = candidate_best_distance(
        candidate
    )

    return (
        -len(families),
        -weighted_score,
        -len(phrases),
        -len(candidate.events),
        best_distance,
        candidate.pub_year
        if candidate.pub_year is not None
        else 9999,
        candidate.doc_id,
    )


def select_candidates(
    comparison: dict,
) -> list[Candidate]:
    """
    Select the most convergent candidate passages.

    Candidates are ranked entirely from semantic retrieval evidence.
    Eye vocabulary is cross-referenced after selection and is never used
    as a gate.

    If fewer than CANDIDATE_LIMIT multi-probe candidates exist, the
    remaining places are filled from the strongest single-probe
    candidates.

    The known-invalid 1800-1849 bucket is excluded before selection.
    """

    candidates = build_candidates(
        comparison
    )

    candidates.sort(
        key=candidate_priority
    )

    multi_probe = []

    for candidate in candidates:

        phrases, _ = candidate_query_summary(
            candidate
        )

        if len(phrases) >= MIN_EVENT_QUERY_OVERLAP:
            multi_probe.append(candidate)

    if len(multi_probe) >= CANDIDATE_LIMIT:
        return multi_probe[:CANDIDATE_LIMIT]

    selected = multi_probe[:]

    remaining = [
        candidate
        for candidate in candidates
        if candidate not in selected
    ]

    selected.extend(
        remaining[
            : CANDIDATE_LIMIT - len(selected)
        ]
    )

    return selected


def candidate_context(
    candidate: Candidate,
) -> str:
    """
    Render one consolidated context for a candidate.

    The widest retrieved context is used as the display basis. Literal
    highlighting includes every semantic probe represented anywhere in the
    candidate; eye terms are highlighted separately as post-retrieval
    cross-reference evidence.
    """

    phrases: set[str] = set()

    for event in candidate.events:
        phrases.update(
            event["queries"]
        )

    # Use the event with the earliest token position as the anchor. The
    # contexts overlap when the events belong to the same passage.
    anchor = min(
        candidate.events,
        key=lambda event: event["token_idx"],
    )

    context_tokens = anchor["context_tokens"]

    return format_context(
        context_tokens,
        anchor["token_idx"],
        tuple(sorted(phrases)),
    )


def render_candidate(
    candidate: Candidate,
    number: int,
) -> str:
    """Render one human-readable candidate passage."""

    phrases, families = candidate_query_summary(
        candidate
    )

    weighted_score = candidate_weighted_score(
        candidate
    )

    best_distance = candidate_best_distance(
        candidate
    )

    bodily = candidate_bodily_hits(candidate)
    eyes = candidate_eye_hits(candidate)

    events = []

    for event in sorted(
        candidate.events,
        key=lambda item: item["token_idx"],
    ):

        event_eye_hits = eye_hits(
            event["context_tokens"]
        )

        event_eye_html = (
            ", ".join(
                escape(term)
                for term in event_eye_hits
            )
            if event_eye_hits
            else "—"
        )

        for phrase, result in sorted(
            event["queries"].items(),
            key=lambda item: (
                item[1]["distance"],
                item[0],
            ),
        ):

            events.append(
                f"""
                <tr>
                    <td>{escape(phrase)}</td>
                    <td>{escape(query_family(phrase))}</td>
                    <td>{PROBE_WEIGHTS[phrase]:.1f}</td>
                    <td>{result["rank"]}</td>
                    <td>{result["distance"]:.6f}</td>
                    <td>{event_eye_html}</td>
                    <td>
                        event {event["event_id"]},
                        token {event["token_idx"]}
                    </td>
                </tr>
                """
            )

    bodily_html = (
        "".join(
            f'<span class="bodily-term">{escape(term)}</span>'
            for term in bodily
        )
        if bodily
        else '<span class="muted">none in displayed context</span>'
    )

    eye_html = (
        "".join(
            f'<span class="eye-term">{escape(term)}</span>'
            for term in eyes
        )
        if eyes
        else '<span class="muted">none in candidate contexts</span>'
    )

    return f"""
    <article class="candidate">

        <div class="candidate-heading">

            <div class="candidate-number">
                {number}
            </div>

            <div class="candidate-title">
                <div class="candidate-year">
                    {escape(str(candidate.pub_year or "Unknown year"))}
                </div>

                <div class="candidate-source">
                    {escape(candidate.author or "Unknown author")}
                    —
                    {escape(candidate.title or "Untitled")}
                </div>

                <div class="candidate-id">
                    {escape(candidate.corpus)}
                    /
                    {escape(candidate.doc_id)}
                </div>
            </div>

        </div>

        <div class="evidence-summary">

            <span class="evidence">
                {len(phrases)} probe phrases
            </span>

            <span class="evidence">
                {len(families)} probe families
            </span>

            <span class="evidence weighted">
                weighted probe evidence: {weighted_score:.1f}
            </span>

            <span class="evidence">
                {len(candidate.events)} retrieved events
            </span>

            <span class="evidence">
                best distance: {best_distance:.6f}
            </span>

        </div>

        <div class="family-list">
            {" ".join(
                f'<span class="family">{escape(family)}</span>'
                for family in sorted(families)
            )}
        </div>

        <div class="context">
            {candidate_context(candidate)}
        </div>

        <div class="cross-reference">

            <div>
                <strong>Eye cross-reference:</strong>
                {eye_html}
            </div>

            <div>
                <strong>Bodily vocabulary:</strong>
                {bodily_html}
            </div>

        </div>

        <details class="retrieval-detail">

            <summary>
                Retrieval evidence
                <span class="count">
                    {len(events)} query/event matches
                </span>
            </summary>

            <table>

                <thead>
                    <tr>
                        <th>Probe</th>
                        <th>Family</th>
                        <th>Weight</th>
                        <th>Rank</th>
                        <th>Distance</th>
                        <th>Eye terms in context</th>
                        <th>Event</th>
                    </tr>
                </thead>

                <tbody>
                    {"".join(events)}
                </tbody>

            </table>

        </details>

    </article>
    """


def render_candidates(
    candidates: list[Candidate],
) -> str:
    """Render the selected candidate set."""

    if not candidates:

        return """
        <p class="empty">
            No candidate passages were generated.
        </p>
        """

    return "".join(
        render_candidate(
            candidate,
            number,
        )
        for number, candidate in enumerate(
            candidates,
            start=1,
        )
    )


def render_raw_results(
    raw_results: dict,
) -> str:
    """Render complete raw retrieval data for audit and exploration."""

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
                            <span class="rank">
                                #{result["rank"]}
                            </span>

                            <span class="distance">
                                distance={result["distance"]:.6f}
                            </span>

                            <span class="year">
                                {escape(
                                    str(
                                        event["pub_year"]
                                        or "Unknown"
                                    )
                                )}
                            </span>

                            <span class="event">
                                event={event["event_id"]}
                            </span>
                        </div>

                        <div class="metadata">

                            <div>
                                <strong>Author</strong>
                                {escape(
                                    event["author"]
                                    or "Unknown"
                                )}
                            </div>

                            <div>
                                <strong>Title</strong>
                                {escape(
                                    event["title"]
                                    or "Untitled"
                                )}
                            </div>

                            <div>
                                <strong>Corpus / document</strong>
                                {escape(event["corpus"])}
                                /
                                {escape(event["doc_id"])}
                            </div>

                            <div>
                                <strong>Retrieved token</strong>
                                {escape(event["token"])}
                                &nbsp;
                                index {event["token_idx"]}
                            </div>

                        </div>

                        <div class="context">
                            {result["context"]}
                        </div>

                    </article>
                    """
                )

            invalid_class = (
                " invalid-bucket"
                if (
                    bucket_start,
                    bucket_end,
                ) in INVALID_CANDIDATE_BUCKETS
                else ""
            )

            invalid_notice = (
                """
                <div class="warning">
                    This chronological bucket is currently excluded from
                    candidate selection because its Lance retrievals are
                    known to be invalid.
                </div>
                """
                if (
                    bucket_start,
                    bucket_end,
                ) in INVALID_CANDIDATE_BUCKETS
                else ""
            )

            bucket_sections.append(
                f"""
                <details class="bucket{invalid_class}">

                    <summary>
                        {bucket_start}–{bucket_end}
                        <span class="count">
                            {len(results)} results
                        </span>
                    </summary>

                    {invalid_notice}

                    {"".join(rows)}

                </details>
                """
            )

        sections.append(
            f"""
            <details class="query">

                <summary>
                    <span class="query-name">
                        {escape(phrase)}
                    </span>

                    <span class="probe-weight">
                        weight={PROBE_WEIGHTS[phrase]:.1f}
                    </span>

                    <span class="count">
                        {sum(
                            len(items)
                            for items in buckets.values()
                        )} results
                    </span>
                </summary>

                {"".join(bucket_sections)}

            </details>
            """
        )

    return "".join(sections)


def render_report(
    *,
    comparison: dict,
    raw_results: dict,
    candidates: list[Candidate],
) -> str:
    """Build the complete self-contained candidate report."""

    event_count = len(comparison)

    multi_probe_events = sum(
        1
        for record in comparison.values()
        if len(record["queries"]) >= MIN_EVENT_QUERY_OVERLAP
        and (
            record["bucket_start"],
            record["bucket_end"],
        ) not in INVALID_CANDIDATE_BUCKETS
    )

    excluded_event_count = sum(
        1
        for record in comparison.values()
        if (
            record["bucket_start"],
            record["bucket_end"],
        ) in INVALID_CANDIDATE_BUCKETS
    )

    eye_cross_reference_count = sum(
        bool(
            candidate_eye_hits(candidate)
        )
        for candidate in candidates
    )

    return f"""<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
    MacBERTh candidate selection
</title>

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
    --eye: rgba(120, 190, 150, 0.32);

    --family: #31452f;
    --family-text: #b9d8b3;

    --warning: #5a4625;
    --warning-text: #f1d59b;
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

.subtitle {{
    color: var(--muted);
    margin-bottom: 24px;
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

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(180px, 1fr)
        );

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

.major-section {{
    margin-top: 16px;
}}

.bucket {{
    margin: 8px;
    background: var(--panel-alt);
}}

.query {{
    margin-top: 10px;
}}

.query-name {{
    color: var(--accent);
}}

.probe-weight {{
    margin-left: 12px;

    color: var(--muted);

    font-family:
        ui-monospace,
        SFMono-Regular,
        Consolas,
        monospace;

    font-size: 0.85rem;
}}

.count {{
    float: right;
    color: var(--muted);
    font-weight: normal;
}}

.candidate {{
    background: var(--panel);

    border: 1px solid var(--border);
    border-radius: 9px;

    padding: 20px;

    margin: 18px 0;
}}

.candidate-heading {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
}}

.candidate-number {{
    min-width: 42px;
    height: 42px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: var(--accent-soft);
    color: var(--accent);

    border-radius: 50%;

    font-weight: 700;
}}

.candidate-year {{
    color: var(--accent);
    font-size: 1.25rem;
    font-weight: 700;
}}

.candidate-source {{
    font-size: 1.05rem;
}}

.candidate-id {{
    margin-top: 3px;

    color: var(--muted);

    font-family:
        ui-monospace,
        SFMono-Regular,
        Consolas,
        monospace;

    font-size: 0.85rem;
}}

.evidence-summary {{
    display: flex;
    flex-wrap: wrap;

    gap: 8px;

    margin: 16px 0 10px;
}}

.evidence {{
    background: var(--accent-soft);
    color: var(--accent);

    border-radius: 5px;

    padding: 4px 9px;

    font-size: 0.88rem;
}}

.evidence.weighted {{
    font-weight: 700;
}}

.family-list {{
    display: flex;
    flex-wrap: wrap;

    gap: 7px;

    margin-bottom: 14px;
}}

.family {{
    background: var(--family);
    color: var(--family-text);

    border-radius: 5px;

    padding: 3px 8px;

    font-size: 0.86rem;
}}

.context {{
    padding: 16px;

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

.eye-match {{
    background: var(--eye);

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

.cross-reference {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(300px, 1fr)
        );

    gap: 10px;

    margin-top: 12px;

    padding: 10px 12px;

    background: var(--panel-alt);

    border-radius: 6px;

    color: var(--muted);

    font-size: 0.9rem;
}}

.eye-term {{
    display: inline-block;

    margin-left: 6px;

    color: var(--text);

    background: var(--eye);

    border-radius: 4px;

    padding: 1px 5px;
}}

.bodily-term {{
    display: inline-block;

    margin-left: 6px;

    color: var(--text);
}}

.muted {{
    color: var(--muted);
}}

.retrieval-detail {{
    margin-top: 16px;

    background: var(--panel-alt);
}}

table {{
    width: 100%;
    border-collapse: collapse;

    margin-top: 12px;

    font-size: 0.92rem;
}}

th,
td {{
    text-align: left;

    padding: 8px 10px;

    border-bottom:
        1px solid var(--border);
}}

th {{
    color: var(--muted);
    font-weight: 600;
}}

.result {{
    margin: 10px;

    padding: 16px;

    border:
        1px solid var(--border);

    border-radius: 7px;

    background: var(--panel);
}}

.result-header {{
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

    font-family:
        ui-monospace,
        SFMono-Regular,
        Consolas,
        monospace;

    font-size: 0.9rem;
}}

.metadata {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(260px, 1fr)
        );

    gap: 6px 24px;

    color: var(--muted);

    font-size: 0.9rem;

    margin-bottom: 14px;
}}

.metadata strong {{
    color: var(--text);
}}

.warning {{
    margin: 10px;

    padding: 10px 14px;

    background: var(--warning);

    color: var(--warning-text);

    border-radius: 6px;

    font-size: 0.9rem;
}}

.invalid-bucket {{
    border-color: var(--warning);
}}

.legend {{
    display: flex;
    flex-wrap: wrap;

    gap: 18px;

    margin: 16px 0;

    color: var(--muted);

    font-size: 0.9rem;
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

    border-top:
        1px solid var(--border);

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

    .candidate-heading {{
        gap: 10px;
    }}

}}

</style>

</head>

<body>

<main>

<h1>
    MacBERTh white-hair semantic search
</h1>

<div class="subtitle">
    Semantic retrieval for white hair and white beard, followed by
    post-retrieval cross-reference for historical eye vocabulary.
</div>

<nav class="toc" aria-label="Table of contents">

    <div class="toc-title">
        Contents
    </div>

    <a href="#overview">
        Overview
    </a>

    <a href="#candidates">
        Candidate passages
    </a>

    <a href="#raw-results">
        Raw retrieval
    </a>

</nav>

<section id="overview">

<h2>
    Overview
</h2>

<div class="config">

    <div class="stat">
        <div class="stat-label">
            Corpus years
        </div>
        <div class="stat-value">
            {MIN_YEAR}–{MAX_YEAR}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Semantic scale
        </div>
        <div class="stat-value">
            {escape(SCALE)}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Semantic probes
        </div>
        <div class="stat-value">
            {len(PHRASES)}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Retrieved events
        </div>
        <div class="stat-value">
            {event_count}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Multi-probe events
        </div>
        <div class="stat-value">
            {multi_probe_events}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Invalid-bucket events
        </div>
        <div class="stat-value">
            {excluded_event_count}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Reading candidates
        </div>
        <div class="stat-value">
            {len(candidates)}
        </div>
    </div>

    <div class="stat">
        <div class="stat-label">
            Candidates with eye evidence
        </div>
        <div class="stat-value">
            {eye_cross_reference_count}
        </div>
    </div>

</div>

<p>
    Semantic probes:
    <code>white hair</code>
    and
    <code>white beard</code>.
</p>

<p>
    Eye cross-reference:
    <code>
        eye, eyes, eyne, eie, eien
    </code>.
</p>

<p>
    Carrier:
    <code>{escape(CARRIER)}</code>
</p>

<div class="legend">

    <span>
        <span class="retrieved">
            retrieved corpus token
        </span>
    </span>

    <span>
        <span class="query-match">
            literal semantic-query term
        </span>
    </span>

    <span>
        <span class="eye-match">
            eye cross-reference
        </span>
    </span>

</div>

<p class="subtitle">
    The eye vocabulary is not used to retrieve, filter, score, or rank
    candidates. It is measured only after the white-hair semantic search
    so that the search asks what resembles white hair or white beard first,
    and then asks whether those passages also mention eyes.
</p>

<p class="subtitle">
    Probe weights affect only candidate reading priority. They do not alter
    MacBERTh vectors or LanceDB distances. The candidate selector does not
    determine historical meaning. It only reduces the number of passages
    requiring human inspection.
</p>

</section>

<section id="candidates">

<h2>
    Candidate passages
</h2>

<p class="subtitle">
    The passages below are the first reading set. A candidate can contain
    several independently retrieved events from the same nearby passage.
    Eye vocabulary is reported separately as post-retrieval evidence.
</p>

{render_candidates(candidates)}

</section>

<section id="raw-results">

<h2>
    Raw retrieval
</h2>

<details class="major-section">

    <summary>
        Complete semantic retrieval results
        <span class="count">
            {len(PHRASES)} probes
        </span>
    </summary>

    <p class="subtitle">
        This section is retained for auditability and for returning to the
        underlying retrieval results when a candidate needs investigation.
        The known-invalid 1800–1849 bucket remains visible here but is
        excluded from candidate selection.
    </p>

    {render_raw_results(raw_results)}

</details>

</section>

<footer>

Generated by the MacBERTh semantic candidate-selection probe.

No PostgreSQL data was modified.

</footer>

</main>

</body>

</html>
"""


def main() -> None:

    validate_probe_configuration()

    logger.info(
        "[phrase-probe] Lance root=%s",
        LANCE_INDEXES_DIR,
    )

    logger.info(
        "[phrase-probe] configured probes=%d",
        len(PHRASES),
    )

    logger.info(
        "[phrase-probe] eye cross-reference terms=%s",
        sorted(EYE_TERMS),
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
        application_name="semantic-phrase-candidate-probe",
    ) as connection:

        for phrase in PHRASES:

            logger.info(
                "[phrase-probe] searching %r weight=%.1f",
                phrase,
                PROBE_WEIGHTS[phrase],
            )

            search_phrase(
                phrase=phrase,
                encoder=encoder,
                store=store,
                connection=connection,
                comparison=comparison,
                raw_results=raw_results,
            )

    candidates = select_candidates(
        comparison
    )

    report = render_report(
        comparison=comparison,
        raw_results=raw_results,
        candidates=candidates,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    multi_probe_events = sum(
        1
        for record in comparison.values()
        if len(record["queries"]) >= MIN_EVENT_QUERY_OVERLAP
        and (
            record["bucket_start"],
            record["bucket_end"],
        ) not in INVALID_CANDIDATE_BUCKETS
    )

    excluded_event_count = sum(
        1
        for record in comparison.values()
        if (
            record["bucket_start"],
            record["bucket_end"],
        ) in INVALID_CANDIDATE_BUCKETS
    )

    eye_cross_reference_count = sum(
        bool(
            candidate_eye_hits(candidate)
        )
        for candidate in candidates
    )

    logger.info(
        "[phrase-probe] retrieved events=%d",
        len(comparison),
    )

    logger.info(
        "[phrase-probe] multi-probe events=%d",
        multi_probe_events,
    )

    logger.info(
        "[phrase-probe] invalid-bucket events=%d",
        excluded_event_count,
    )

    logger.info(
        "[phrase-probe] reading candidates=%d",
        len(candidates),
    )

    logger.info(
        "[phrase-probe] candidates with eye evidence=%d",
        eye_cross_reference_count,
    )

    logger.info(
        "[phrase-probe] HTML report=%s",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
