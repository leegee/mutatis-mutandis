#!/usr/bin/env python
"""
label_clusters.py - generate semantic sense labels for concept clusters.

Replaces the manual points -> rawText -> Groq flow (cluster2groq.ts /
groq-middleware.ts) with a data-driven pipeline that:

  1. Reads cluster membership from SQLite (events, concept_cluster_info,
     concept_aggregate).
  2. Diversifies the sample per cluster by capping occurrences per doc_id,
     so a cluster that is actually "one document said this word 56 times"
     (see e.g. LIBERTY cluster H, 56/56 points from doc A89452) cannot
     dominate the sample the LLM sees.
  3. Pulls real surrounding-context text for the sampled events from
     Postgres (NOT bare tokens or doc_id counts - see conversation
     history for why token/doc aggregates alone are nearly useless for
     concepts whose anchor token barely varies).
  4. Dedupes near-identical context lines (character-normalised, same
     idea as the TS dedup step) before sending to the LLM.
  5. Calls Groq once per (concept, cluster_id), in parallel-safe fashion
     (cluster_id is threaded through the whole pipeline, not inferred
     from call order).
  6. Writes sense_name / sense_description back into concept_cluster_info.

ASSUMPTION FLAGGED FOR REVIEW:
    fetch_window_text() assumes a Postgres table `pamphlet_corpus` with a
    `full_text` column, and reconstructs a window by splitting on
    whitespace and slicing around `token_idx`. This is almost certainly
    NOT your real schema - swap this function's body for whatever your
    actual text storage looks like (a pre-tokenized table, a windows
    table keyed by window_id, etc). Everything else in this script is
    independent of that choice.
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import CORPUS_TIER2_DB_PATH

try:
    from groq import Groq
except ImportError:
    Groq = None  # allow --dry-run without the groq package installed

import os

GROQ_MODEL = "llama-3.3-70b-versatile"

MIN_LABEL_CLUSTER_SIZE = 20   # Limit LLM calls
MAX_LLM_CONCENTRATION = 0.8   # Try to avoid dominance by one doc
MAX_EVENTS_PER_DOC = 2        # cap per document, so no single text dominates
SAMPLE_SIZE_PER_CLUSTER = 8   # total events sampled per cluster, post-cap
CONTEXT_WINDOW_TOKENS = 15    # words either side of the anchor token
MAX_PROMPT_CHARS = 3000       # truncate assembled context before sending

# concentration threshold: if the single most-common doc_id (or token,
# though token barely varies by design) accounts for
# this fraction or more of a cluster's raw points, flag it rather than
# silently labeling it as if it were a real semantic grouping.
DEGENERATE_CONCENTRATION_THRESHOLD = 0.6


@dataclass
class ClusterEvent:
    event_id: int
    doc_id: str
    token_idx: int
    window_id: Optional[int]
    token: Optional[str]
    pub_year: Optional[int]


@dataclass
class ClusterSample:
    concept: str
    cluster_id: int
    point_count: int
    concentration: float
    dominant_doc_id: Optional[str]
    cluster_label: Optional[str] = None
    historical_start: Optional[int] = None
    historical_end: Optional[int] = None
    events: list[ClusterEvent] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)


#  SQLite side

def sqlite_cx(db_path=None):
    dbh = sqlite3.connect(db_path or CORPUS_TIER2_DB_PATH)
    dbh.row_factory = sqlite3.Row
    return dbh


def fetch_clusters_for_concept(con, concept: str) -> list[dict]:
    rows = con.execute(
        """
        SELECT cluster_id, cluster_label, point_count
        FROM concept_cluster_info
        WHERE concept = ?
        ORDER BY point_count DESC
        """,
        (concept,),
    ).fetchall()

    return [dict(r) for r in rows]


def fetch_cluster_events(con, concept: str, cluster_id: int) -> list[ClusterEvent]:
    """Raw event membership for one cluster - the full point set, not yet sampled."""
    rows = con.execute(
        """
        SELECT event_id,
            doc_id,
            token_idx,
            window_id,
            token,
            pub_year
        FROM events
        WHERE concept = ?
        AND cluster_id = ?
        """,
        (concept, cluster_id),
    ).fetchall()
    return [
        ClusterEvent(
            event_id=r["event_id"],
            doc_id=r["doc_id"],
            token_idx=r["token_idx"],
            window_id=r["window_id"],
            token=r["token"],
            pub_year=r["pub_year"],
        )
        for r in rows
    ]


def compute_concentration(events: list[ClusterEvent]) -> tuple[float, Optional[str]]:
    """
    Fraction of a cluster's points coming from its single most common
    doc_id, and which doc_id that is. High concentration is the signature
    of the degenerate "one document repeats this word" cluster identified
    manually in the LIBERTY sample (clusters H/I: 56/56 and 12/12 points
    from one doc each).
    """
    if not events:
        return 0.0, None
    counts = defaultdict(int)
    for e in events:
        counts[e.doc_id] += 1
    dominant_doc_id, top_count = max(counts.items(), key=lambda kv: kv[1])
    return top_count / len(events), dominant_doc_id


def diversify_sample(
    events: list[ClusterEvent],
    max_per_doc: int = MAX_EVENTS_PER_DOC,
    sample_size: int = SAMPLE_SIZE_PER_CLUSTER,
) -> list[ClusterEvent]:
    """
    Cap occurrences per doc_id, then take up to sample_size events,
    preferring breadth across documents over raw recall. This is what
    actually fixes the concentration problem - a 56/56-single-doc cluster
    yields at most `max_per_doc` sampled events here, not 56.
    """
    per_doc_seen: dict[str, int] = defaultdict(int)
    diversified = []
    for e in events:
        if per_doc_seen[e.doc_id] >= max_per_doc:
            continue
        per_doc_seen[e.doc_id] += 1
        diversified.append(e)
        if len(diversified) >= sample_size:
            break
    return diversified


def write_label_to_sqlite(
    db_path,
    concept: str,
    cluster_id: int,
    sense_name: str,
    description: str,
    model: str,
    prompt: str,
    sample_event_ids: list[int],
    concentration: float,
):
    con = sqlite3.connect(db_path or CORPUS_TIER2_DB_PATH)

    con.execute(
        """
        UPDATE concept_cluster_info
        SET
            cluster_label = ?,
            description = ?,
            llm_model = ?,
            llm_concentration = ?,
            llm_prompt = ?,
            llm_timestamp = datetime('now'),
            llm_sample_size = ?,
            llm_sample_event_ids = ?
        WHERE concept = ?
        AND cluster_id = ?
        """,
        (
            sense_name,
            description,
            model,
            concentration,
            prompt,
            len(sample_event_ids),
            json.dumps(sample_event_ids),
            concept,
            cluster_id,
        ),
    )

    con.commit()
    con.close()


#  Postgres side
def fetch_window_text(
    pg_dbh,
    doc_id: str,
    token_idx: int,
    window_tokens: int = CONTEXT_WINDOW_TOKENS,
) -> Optional[str]:
    with pg_dbh.cursor() as cur:
        cur.execute(
            """
            SELECT token, token_idx
            FROM pamphlet_tokens
            WHERE doc_id = %s
              AND token_idx BETWEEN %s AND %s
            ORDER BY token_idx
            """,
            (
                doc_id,
                token_idx - window_tokens,
                token_idx + window_tokens,
            ),
        )

        rows = cur.fetchall()

    content = " ".join(
        f"<mark>{token}</mark>" if idx == token_idx else token
        for token, idx in rows
    )

    content = re.sub(r"\s+([,.;:\)])", r"\1", content)
    content = re.sub(r"\(\s+", "(", content)

    return content


#  text cleanup

def dedup_lines(lines: list[str]) -> list[str]:
    """
    Drop near-duplicate context lines by normalised (lowercased,
    punctuation/whitespace-stripped) comparison. Mirrors the TS dedup
    step. Note this only catches literal near-duplicates - it does NOT
    fix concentration (many distinct sentences from the same document
    won't match each other here); diversify_sample() is what fixes that,
    upstream of this step.
    """
    seen = set()
    result = []
    for line in lines:
        if not line:
            continue
        normalised = re.sub(r"[\s\W]+", "", line.lower(), flags=re.UNICODE)
        if normalised in seen:
            continue
        seen.add(normalised)
        result.append(line)
    return result


# Assemble and call LLM

def build_cluster_sample(
    sqlite_dbh, pg_dbh, concept: str, cluster_info: dict
) -> ClusterSample:
    cluster_id = cluster_info["cluster_id"]
    events = fetch_cluster_events(sqlite_dbh, concept, cluster_id)
    concentration, dominant_doc_id = compute_concentration(events)

    years = [
        e.pub_year
        for e in events
        if e.pub_year is not None
    ]
    historical_start = min(years) if years else None
    historical_end = max(years) if years else None

    events = sorted(
        events,
        key=lambda e: (
            e.pub_year or 9999,
            e.doc_id,
            e.token_idx
        )
    )

    sampled = diversify_sample(events)

    context_lines = []
    for e in sampled:
        text = fetch_window_text(pg_dbh, e.doc_id, e.token_idx)
        if text:
            context_lines.append(text)

    context_lines = dedup_lines(context_lines)

    return ClusterSample(
        concept=concept,
        cluster_id=cluster_id,
        cluster_label=cluster_info.get("cluster_label"),
        point_count=cluster_info["point_count"],
        concentration=concentration,
        dominant_doc_id=dominant_doc_id,
        historical_start=historical_start,
        historical_end=historical_end,
        events=sampled,
        context_lines=context_lines,
    )


def build_prompt(sample: ClusterSample) -> str:
    joined = "\n\n".join(sample.context_lines)[:MAX_PROMPT_CHARS]

    if sample.historical_start and sample.historical_end:
        historical_span = (
            f"{sample.historical_start}-{sample.historical_end}"
        )
    else:
        historical_span = "unknown"

    return f"""
Concept: {sample.concept}
Cluster size: {sample.point_count} occurrences
Historical span: {historical_span}

Representative occurrences:

{joined}
"""


def call_groq(client, sample: ClusterSample) -> dict:
    prompt_body = build_prompt(sample)

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in historical English semantics, "
                    "specializing in Early English Books Online (EEBO). "
                    "You label semantic usage clusters based on contextual evidence."
                ),
            },
            {
                "role": "user",
                "content": f"""
TASK:
Given a cluster of occurrences of a word, identify ONE unified semantic sense.

CLUSTER DATA:
{prompt_body}

RULES:
- Only ONE sense
- Do NOT output multiple meanings
- Do NOT give dictionary definitions
- Do NOT provide historical commentary
- Focus only on usage in this cluster

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "sense_name": string (max 8 words),
  "description": string (1-2 sentences)
}}
""",
            },
        ],
        model=GROQ_MODEL,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    if not raw:
        raise RuntimeError(f"No model output for {sample.concept} cluster {sample.cluster_id}")

    return safe_json_parse(raw)


def safe_json_parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise RuntimeError(f"No JSON found in model output: {raw[:200]}")
        return json.loads(match.group(0))


# Runners

def label_concept_clusters(concept: str, db_path=None, dry_run: bool = False):
    sqlite_dbh = sqlite_cx(db_path)
    pg_dbh = get_connection()

    client = None
    if not dry_run:
        if Groq is None:
            raise RuntimeError("groq package not installed - pip install groq, or pass --dry-run")
        client = Groq(api_key=os.environ["GROQ_API_KEY"])

    try:
        clusters = fetch_clusters_for_concept(sqlite_dbh, concept)
        clusters = [
            c for c in clusters
            if c["point_count"] >= MIN_LABEL_CLUSTER_SIZE
        ]
        logger.info(f"[label_clusters] {concept}: {len(clusters)} clusters")

        for cluster_info in clusters:
            sample = build_cluster_sample(sqlite_dbh, pg_dbh, concept, cluster_info)

            if sample.concentration >= MAX_LLM_CONCENTRATION:
                logger.warning(
                    f"[label_clusters] skipping document-dominated cluster "
                    f"{concept} {sample.cluster_id}"
                )
                continue

            if not sample.context_lines:
                logger.warning(
                    f"[label_clusters] {concept} cluster {sample.cluster_id}: "
                    f"no context text retrieved, skipping"
                )
                continue

            if sample.concentration >= DEGENERATE_CONCENTRATION_THRESHOLD:
                logger.warning(
                    f"[label_clusters] {concept} cluster {sample.cluster_id}: "
                    f"{sample.concentration:.0%} of points from doc_id={sample.dominant_doc_id} "
                    f"({sample.point_count} total points) - likely a single-document "
                    f"repetition artifact, not a genuine semantic cluster. "
                    f"Labeling anyway from the diversified sample, but flag for review."
                )

            if dry_run:
                logger.info(
                    f"[label_clusters] DRY RUN {concept} cluster {sample.cluster_id}: "
                    f"would send {len(sample.context_lines)} context lines "
                    f"(concentration={sample.concentration:.0%})"
                )
                continue

            result = call_groq(client, sample)
            logger.info(
                f"[label_clusters] {concept} cluster {sample.cluster_id} -> "
                f"{result.get('sense_name')!r}"
            )

            write_label_to_sqlite(
                db_path,
                concept,
                sample.cluster_id,
                result.get("sense_name", ""),
                result.get("description", ""),
                GROQ_MODEL,
                build_prompt(sample),
                [e.event_id for e in sample.events],
                sample.concentration,
            )

    finally:
        sqlite_dbh.close()
        pg_dbh.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept")
    parser.add_argument("--db-path", default=CORPUS_TIER2_DB_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip the Groq API call; just show what would be sent per cluster",
    )
    args = parser.parse_args()

    if args.concept:
        label_concept_clusters(args.concept, db_path=args.db_path, dry_run=args.dry_run)
    else:
        sqlite_dbh = sqlite_cx(args.db_path)
        concepts = [
            row[0]
            for row in sqlite_dbh.execute("SELECT DISTINCT concept FROM concepts ORDER BY concept")
        ]
        sqlite_dbh.close()
        for concept in concepts:
            label_concept_clusters(concept, db_path=args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
