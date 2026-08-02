#!/usr/bin/env python
"""
experiments/dss-concept-sliced.py

Observation-level semantic backcasting/forecasting.

Given occurrences of a literal phrase (a lexical stand-in for a q --
matched exactly and case-insensitively against corpus text, not resolved
via any curated q tagging) in a source period (`--source_start`/
`--source_end`), finds contextual observations in a comparison period
(`--compare_start`/`--compare_end`) that occupy similar semantic positions.

This is a structural/semantic-proximity comparison over time, not a prediction
or causal claim.

The two periods must not overlap, so the inferred label is always well-defined.

No centroids.
No clustering.
No dimensionality reduction.

Four entry points, one shared core:

  core()    -- pure computation, no side effects. Takes an already-built
               `lookup` (required) and optional `conn`/`faiss_index` to
               reuse; creates nothing, writes nothing, prints nothing.
               Raises ValueError on bad input (overlapping periods, zero
               source observations). Everything below calls this.

  run()     -- one-shot CLI/notebook use. Builds `lookup` (and opens a
               DB connection per phrase search) itself if not given one,
               calls core(), writes the result to `--output`, optionally
               prints a table. Lets ValueError propagate.

  service() -- persistent-process use. Takes `conn`, `lookup`, and
               `faiss_index` as *required* arguments -- all built once by
               the caller at process startup and reused across every
               call -- and just calls core() and returns the result
               dict. No file writing, no printing, no protocol handling
               (HTTP/WS/CGI/etc. is the caller's problem). Lets
               ValueError propagate for the caller's protocol layer to
               translate into an error response.

  main()    -- CLI wrapper. Parses sys.argv, calls run(), turns
               ValueError into a parser.error() exit.

TODO
----
Next steps:

- Instead of immediately aggregating to tokens, keep the matched events. Eg:

    matched_event = {
        "event_id": eid,
        "score": data["rrf_score"],
        "year": year,
        "doc_id": ...,
        "token": ...,
        "position": ...,
    }

- Retrieve context from the observation positions so we have the literal content.

- Graph: instead of summing token/weight, build matched event -> kNN -> co-occurence graph.

- Record distribution over docs.

- Cluster kNN

"""

from __future__ import annotations

import argparse
import json
import math

from lib.corpus_config import (
    CORPUS_MAX_YEAR,
    ZARR_PATH,
    TMP_DIR
)

from lib.corpus_faiss import CorpusFaissIndex
from lib.corpus_logging import logger
from lib.zarr_event_lookup import ZarrEventLookup
from lib.corpus_db import get_connection


# --- copied from experiments/corpus_phrase_search.py -----------------------
#
# corpus_phrase_search.py is an experimental script, not a lib, so there's
# nothing to import here -- these are pasted in verbatim (minus the
# diagnostic/CLI-only pieces this script doesn't need: inspect_missing_
# occurrences() and main()). Keep in sync with corpus_phrase_search.py by
# hand if that script changes.

def normalise_phrase(phrase: str) -> list[str]:
    """
    Convert the user query into the exact lexical tokens used for matching.
    """
    tokens = phrase.split()
    if not tokens:
        raise ValueError("Phrase must contain at least one token")
    return [token.lower() for token in tokens]


def find_phrase_occurrences(
    conn,
    phrase: list[str],
    corpus: str | None = None,
):
    """
    Find exact consecutive token sequences in the corpus.

    Matching is performed on token_idx rather than textual reconstruction so
    corpus tokenisation remains authoritative.
    """
    aliases = [f"t{i}" for i in range(len(phrase))]

    joins = []
    where = []
    params: list[object] = []

    for i, token in enumerate(phrase):
        alias = aliases[i]

        if i > 0:
            previous = aliases[i - 1]

            joins.append(
                f"""
                JOIN pamphlet_tokens {alias}
                  ON {alias}.corpus = {previous}.corpus
                 AND {alias}.doc_id = {previous}.doc_id
                 AND {alias}.token_idx = {previous}.token_idx + 1
                """
            )

        where.append(f"LOWER({alias}.token) = %s")
        params.append(token)

    if corpus is not None:
        where.append("t0.corpus = %s")
        params.append(corpus)

    query = f"""
        SELECT
            t0.corpus,
            t0.doc_id,
            t0.token_idx
        FROM pamphlet_tokens t0
        {" ".join(joins)}
        WHERE {" AND ".join(where)}
        ORDER BY
            t0.corpus,
            t0.doc_id,
            t0.token_idx
    """

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def resolve_tier1_events(lookup, occurrences):
    """
    Resolve corpus occurrences to Tier 1 observation event IDs.

    A corpus occurrence may have multiple Tier 1 observations because the
    same token can be represented under multiple contextual windows.

    Missing Tier 1 observations are reported rather than discarded.
    """
    event_ids_by_position = lookup.find_event_ids_by_positions(
        occurrences
    )

    resolved = []
    missing = []

    for corpus, doc_id, token_idx in occurrences:
        key = (
            str(corpus),
            str(doc_id),
            int(token_idx),
        )

        event_ids = event_ids_by_position.get(key, [])

        if event_ids:
            resolved.append(
                {
                    "corpus": corpus,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                    "event_ids": event_ids,
                }
            )
        else:
            missing.append(
                {
                    "corpus": corpus,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                }
            )

    return resolved, missing


def search_phrase(
    phrase: str,
    *,
    corpus: str | None = None,
    lookup=None,
    conn=None,
):
    """
    Resolve a corpus phrase to Tier 1 event IDs.

    The returned counts deliberately distinguish corpus occurrences,
    matched corpus occurrences, and Tier 1 observation events.

    If `conn` is passed in, it's used as-is and left open -- the caller
    owns its lifecycle (this is the persistent-service path: one
    long-lived connection reused across many calls). If omitted, a
    connection is opened here and closed before returning, exactly as
    before (the one-shot CLI/notebook path).
    """
    tokens = normalise_phrase(phrase)

    owns_conn = conn is None

    if owns_conn:
        conn = get_connection()

    try:
        occurrences = find_phrase_occurrences(
            conn,
            tokens,
            corpus=corpus,
        )
    finally:
        if owns_conn:
            conn.close()

    if lookup is None:
        lookup = ZarrEventLookup(ZARR_PATH)

    resolved, missing = resolve_tier1_events(
        lookup,
        occurrences,
    )

    tier1_event_count = sum(
        len(occurrence["event_ids"])
        for occurrence in resolved
    )

    return {
        "phrase": phrase,
        "tokens": tokens,
        "corpus": corpus,
        "occurrences": len(occurrences),
        "matched": len(resolved),
        "missing": len(missing),
        "tier1_events": tier1_event_count,
        "matches": resolved,
        "missing_occurrences": missing,
    }

# --- end copied from experiments/corpus_phrase_search.py -------------------


def load_field_events(lookup, q, start_year, end_year, conn=None):
    """
    Resolve a q to Tier 1 observation events via corpus phrase search.

    `q` is treated as the lexical phrase to match against the corpus
    (case-insensitive, consecutive tokens -- see search_phrase() above).
    Every corpus occurrence can resolve to multiple Tier 1 events; those are
    deduplicated by event_id and then filtered to [start_year, end_year].

    `conn`, if given, is passed straight through to search_phrase() and
    reused rather than opened fresh -- see search_phrase()'s docstring.
    """
    result = search_phrase(q, lookup=lookup, conn=conn)

    logger.info(
        f"[dss] q={q!r} phrase search: "
        f"corpus_occurrences={result['occurrences']} "
        f"matched={result['matched']} "
        f"missing={result['missing']} "
        f"tier1_events={result['tier1_events']}"
    )

    if result["occurrences"] == 0:
        logger.info(
            f"[dss] q={q!r} does not appear as a literal "
            f"corpus phrase -- check spelling/casing/tokenisation "
            f"(matching is case-insensitive but exact, not stemmed)"
        )

    events = []
    seen_ids = set()

    for match in result["matches"]:
        for eid in match["event_ids"]:
            eid = int(eid)

            if eid in seen_ids:
                continue

            seen_ids.add(eid)

            pos = lookup.get_pos(eid)
            pub_year = int(lookup.pub_year[pos])

            if start_year <= pub_year <= end_year:
                events.append((eid, pub_year))

    events.sort(key=lambda x: (x[1], x[0]))

    return events


def load_indexes(start_year, end_year):
    """
    Disk-backed load of the per-year comparison indices for
    [start_year, end_year]. Used by the one-shot CLI/notebook path
    (run(), with no faiss_index supplied): fine for a single run, but
    re-reads from disk on every call, which is wasteful for a persistent
    service handling many requests against overlapping year ranges --
    see slice_preloaded_index() below for that path instead.
    """
    return CorpusFaissIndex.load_existing_range(
        start_year=start_year,
        end_year=end_year,
        workers=8,
    )


def slice_preloaded_index(faiss_index, start_year, end_year):
    """
    Extract the per-year comparison-index mapping for [start_year,
    end_year] out of a CorpusFaissIndex instance that has *already*
    loaded every year (e.g. via CorpusFaissIndex.load_all()), instead of
    hitting disk again the way load_indexes() does. This is what lets
    service() answer repeated requests without re-reading FAISS indices
    off disk per request.

    ASSUMPTION, please verify against the real lib.corpus_faiss module
    (not visible from this script): this expects `faiss_index` to expose
    its loaded per-year data as `faiss_index.indexes`, a mapping of
    {year: {scale: <per-scale searchable index>}} -- the same shape
    CorpusFaissIndex.load_existing_range() returns and that
    search_historical() already consumes unchanged either way. If the
    real attribute is named differently, or access goes through a method
    rather than a plain attribute, this is the only function that needs
    to change -- core()/search_historical() don't care how the mapping
    was produced.
    """
    return {
        year: scales
        for year, scales in faiss_index.indexes.items()
        if start_year <= year <= end_year
    }


def comparison_slices(start, end, width=10):
    """
    Yield (slice_start, slice_end) year bands spanning [start, end] in
    steps of `width`, used to batch-load per-year FAISS indices for the
    comparison period (whichever period that is -- early or late,
    depending on --direction).
    """
    year = start
    while year <= end:
        yield year, min(year + width - 1, end)
        year += width


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}

    for ranked in ranked_lists.values():
        for rank, (_, eid, year, scale) in enumerate(
            ranked,
            start=1,
        ):
            if eid not in scores:
                scores[eid] = {
                    "rrf_score": 0.0,
                    "year": year,
                    "scale": scale,
                }

            scores[eid]["rrf_score"] += (
                1.0 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True,
    )


def search_historical(historical_index, source_vectors, top_k, exclude_ids):
    """
    Batched per-(scale, year) FAISS search across all source events at once.

    Each (scale, year) pair issues a single index.search() call against the
    full matrix of source vectors for that scale, rather than one call per
    source event. This is the same retrieval as before, restructured so
    FAISS does the batching it's designed for instead of being driven event
    by event.

    This search is direction-agnostic: `historical_index` is just whichever
    set of per-year indices was loaded for the comparison period (early or
    late), and `source_vectors` is just whichever set of event vectors was
    loaded for the source period (late or early respectively). Nothing here
    assumes the comparison period precedes the source period chronologically.

    top_k controls recall depth per (year, scale) index -- how many nearest
    neighbours to pull out of each individual year's index. It does NOT cap
    the final number of candidates returned per source event: the fused
    list below pools every year and scale in the slice together, and is
    returned in full rather than re-truncated to top_k. Truncating it to
    top_k a second time here used to mean a whole slice -- regardless of
    how many years it spanned -- shared one top_k-sized budget of surviving
    candidates, which silently starved most years of any representation
    once results were broken out by individual year. Downstream limits
    (serialise_field's `limit`, print_trajectory's --table_limit) are the
    right place to truncate for display; this stays a recall parameter.

    exclude_ids removes source events from their own matches -- without
    this, a source event can retrieve itself if a comparison slice happens
    to overlap the source period, which would silently dominate the
    resulting field with a spuriously perfect self-match.
    """
    scales = ("local", "medium", "broad")
    n = len(source_vectors["local"])

    per_event_scale_results = [
        {scale: [] for scale in scales}
        for _ in range(n)
    ]

    for scale in scales:
        query_matrix = source_vectors[scale]

        for year, indexes in historical_index.items():
            scores, ids = indexes[scale].search(
                query_matrix,
                top_k,
            )

            for i in range(n):
                for score, eid in zip(scores[i], ids[i]):
                    if eid == -1 or int(eid) in exclude_ids:
                        continue

                    per_event_scale_results[i][scale].append(
                        (
                            float(score),
                            int(eid),
                            year,
                            scale,
                        )
                    )

    results = []

    for i in range(n):
        for scale in scales:
            per_event_scale_results[i][scale].sort(
                key=lambda x: x[0],
                reverse=True,
            )

        fused = reciprocal_rank_fusion(
            per_event_scale_results[i]
        )
        results.append(fused)

    return results


def combine_historical_tokens(matches, lookup):
    """
    Aggregate matched events into a weighted token field.

    weight is total RRF mass contributed by a token across all matched
    events. avg_weight divides that by event count, so a token that is a
    strong match for a few events can be distinguished from one that is a
    mediocre match spread across many -- weight alone conflates the two.
    """
    field = {}

    matched_events = set()
    matched_docs = set()
    matched_years = set()

    for ranked in matches:
        seen = set()

        for eid, data in ranked:
            if eid in seen:
                continue

            seen.add(eid)
            matched_events.add(eid)

            pos = lookup.get_pos(eid)

            doc_id = str(lookup.doc_id[pos])
            matched_docs.add(doc_id)

            year = int(data["year"])
            matched_years.add(year)

            token = str(
                lookup.token[pos]
            ).lower()

            if token not in field:
                field[token] = {
                    "weight": 0.0,
                    "events": 0,
                    "years": set(),
                }

            field[token]["weight"] += data["rrf_score"]
            field[token]["events"] += 1
            field[token]["years"].add(year)

    stats = {
        "matched_events": len(matched_events),
        "matched_documents": len(matched_docs),
        "matched_years": len(matched_years),
        "unique_tokens": len(field),
    }

    field = sorted(
        field.items(),
        key=lambda x: x[1]["weight"],
        reverse=True,
    )

    return field, stats


def combine_historical_tokens_by_year(matches, lookup):
    """
    Same aggregation as combine_historical_tokens(), but split per
    individual publication year of the matched historical event, rather
    than lumped across the whole (possibly --slice_width-years-wide) slice.

    Every match already carries the year of the FAISS index it was
    retrieved from (search_historical tags each candidate with it) -- this
    just groups by that instead of discarding it into one combined field.

    Returns {year: (field, stats)} -- same field/stats shape
    combine_historical_tokens() produces per slice, so serialise_field()
    is reused unchanged for each year.

    Note matched_years and year_coverage are trivially 1 / 1.0 in every
    bucket here, since each bucket *is* a single year by construction --
    those two stats are only informative at multi-year (slice) granularity.
    """
    buckets = {}   # year -> {"field": {token: {...}}, "matched_events": set(), "matched_docs": set()}

    for ranked in matches:
        seen = set()

        for eid, data in ranked:
            if eid in seen:
                continue

            seen.add(eid)

            year = int(data["year"])

            bucket = buckets.setdefault(
                year,
                {
                    "field": {},
                    "matched_events": set(),
                    "matched_docs": set(),
                },
            )

            bucket["matched_events"].add(eid)

            pos = lookup.get_pos(eid)

            doc_id = str(lookup.doc_id[pos])
            bucket["matched_docs"].add(doc_id)

            token = str(
                lookup.token[pos]
            ).lower()

            field = bucket["field"]

            if token not in field:
                field[token] = {
                    "weight": 0.0,
                    "events": 0,
                    "years": {year},
                }

            field[token]["weight"] += data["rrf_score"]
            field[token]["events"] += 1

    results = {}

    for year, bucket in buckets.items():
        field = sorted(
            bucket["field"].items(),
            key=lambda x: x[1]["weight"],
            reverse=True,
        )

        stats = {
            "matched_events": len(bucket["matched_events"]),
            "matched_documents": len(bucket["matched_docs"]),
            "matched_years": 1,
            "unique_tokens": len(field),
        }

        results[year] = (field, stats)

    return results


def serialise_field(field, stats, period_width, limit=100):
    total = sum(
        data["weight"]
        for _, data in field
    )

    entropy = 0.0

    if total:
        for _, data in field:
            p = data["weight"] / total
            entropy -= p * math.log(p)

    return {
        "summary": {
            "entropy": round(entropy, 4),
            "matched_events": stats["matched_events"],
            "matched_documents": stats["matched_documents"],
            "matched_years": stats["matched_years"],
            "year_coverage": round(
                stats["matched_years"] / period_width,
                3,
            ),
            "unique_tokens": stats["unique_tokens"],
        },

        "tokens": [
            {
                "token": token,
                "weight": round(data["weight"], 6),
                "weight_norm": round(
                    data["weight"] / total,
                    8,
                ) if total else 0.0,
                "avg_weight": round(
                    data["weight"] / data["events"],
                    6,
                ) if data["events"] else 0.0,
                "events": data["events"],
                "years": sorted(data["years"]),
            }
            for token, data in field[:limit]
        ],
    }


def normalised_entropy(rows):
    """
    Shannon entropy of weight_norm across tokens in this table, normalised
    to [0, 1] by dividing by log(n_tokens) so periods with different token
    counts are comparable to each other.

    Deliberately distinct from summary["entropy"] as computed by
    serialise_field(), which is raw (unnormalised) entropy over the full
    unlimited field and over `weight` rather than `weight_norm`. Both are
    printed in the period header below.
    """
    weights = [
        row["weight_norm"]
        for row in rows
        if row.get("weight_norm", 0) > 0
    ]

    if len(weights) < 2:
        return 0.0

    total = sum(weights)

    entropy = -sum(
        (w / total) * math.log(w / total)
        for w in weights
    )

    return entropy / math.log(len(weights))


def field_stats(rows):
    weights = [
        row["weight_norm"]
        for row in rows
        if row.get("weight_norm", 0) > 0
    ]

    if not weights:
        return {
            "entropy": 0.0,
            "effective": 0.0,
            "dominance": 0.0,
            "tokens": 0,
        }

    entropy = normalised_entropy(rows)

    return {
        "entropy": entropy,
        "effective": math.exp(entropy),
        "dominance": weights[0],
        "tokens": len(weights),
    }


def format_years(years, max_shown=6):
    if len(years) <= max_shown:
        return years

    half = max_shown // 2
    return years[:half] + ["..."] + years[-half:]


def print_trajectory(q, source, direction, trajectory, limit=5):
    """
    Pretty-print a trajectory dict (same shape written to --output) to the
    terminal, so a run is inspectable without a separate script/re-reading
    the JSON back off disk.
    """
    print()
    print(f"Q: {q}  (source={source}, direction={direction})")
    print("=" * 88)

    for period, field in trajectory.items():
        summary = field.get("summary", {})
        rows = field.get("tokens", [])

        if not rows:
            continue

        rows = sorted(
            rows,
            key=lambda x: x.get(
                "weight_norm",
                x.get(
                    "weight",
                    0,
                ),
            ),
            reverse=True,
        )

        stats = field_stats(rows)

        print()
        print(
            f"PERIOD: {period}"
            f"  H={stats['entropy']:.3f}"
            f"  N={stats['tokens']}"
            f"  eff={stats['effective']:.2f}"
            f"  top={stats['dominance']:.3f}"
        )
        print(
            f"  events={summary.get('matched_events', '?')}"
            f"  docs={summary.get('matched_documents', '?')}"
            f"  years_covered={summary.get('matched_years', '?')}"
            f"  coverage={summary.get('year_coverage', '?')}"
            f"  H_raw={summary.get('entropy', '?')}"
        )
        print("-" * 88)

        for row in rows[:limit]:
            years_display = format_years(row.get("years", []))

            print(
                f"{row['token']:<25}"
                f"{row.get('weight_norm', row.get('weight', 0)):10.4f}"
                f"  n={row.get('events', '?'):<4}"
                f"  years={years_display}"
            )


def core(
    q,
    *,
    source_start,
    source_end,
    compare_start,
    compare_end,
    lookup,
    neighbours=10,
    slice_width=10,
    granularity="year",
    conn=None,
    faiss_index=None,
):
    """
    Run a single backcast/forecast and return the result dict -- q,
    direction, source, compare, trajectory. No side effects: doesn't
    write a file, doesn't print, and doesn't create or tear down any
    expensive resource. Every caller (run(), service(), and indirectly
    main() through run()) funnels through here; this is the one place
    the actual comparison logic lives.

    `lookup` is required and must already have a full-corpus index
    attached (via lookup.attach_index(...)) -- core() never builds one
    itself, since building it is exactly the expensive step callers with
    persistent resources (service()) want to do once, not per call.

    `conn`, if given, is passed through to load_field_events() ->
    search_phrase() and reused as-is rather than opened fresh -- see
    search_phrase()'s docstring. If omitted, a connection is opened and
    closed internally for this call only, matching the original
    one-shot behaviour.

    `faiss_index`, if given, must be a CorpusFaissIndex instance that has
    already loaded every year (e.g. via CorpusFaissIndex.load_all()) --
    each comparison slice is then read out of it in memory via
    slice_preloaded_index() instead of hitting disk. If omitted, each
    slice is loaded from disk on demand via load_indexes(), matching the
    original one-shot behaviour. This is the other half of what makes
    service() cheap to call repeatedly: no per-request disk reads for
    comparison-period indices, same as no per-request DB connection via
    `conn` above.

    Raises ValueError for user-input problems (overlapping periods, zero
    source observations). There's no argparse.Namespace or parser here,
    so this can't call parser.error() (which calls sys.exit() -- fine
    for a CLI process, fatal for a notebook kernel or a persistent
    service handling many independent requests). Each caller decides
    what to do with the exception: main() turns it into parser.error(),
    run() lets it propagate to the notebook/script caller, service()
    lets it propagate to whatever protocol layer is calling it (e.g. to
    become an HTTP 400).
    """
    # The two periods must not overlap -- otherwise a comparison slice can
    # overlap the source period and events can match themselves (this is
    # belt-and-braces on top of the exclude_ids check below, which handles
    # the case at the individual-event level; this handles it at the
    # range level so a mostly-overlapping run fails fast instead of just
    # silently losing most of its candidates to exclusion).
    if source_start <= compare_end and compare_start <= source_end:
        raise ValueError(
            f"source period ({source_start}-{source_end}) and compare "
            f"period ({compare_start}-{compare_end}) must not overlap"
        )

    # Which period is chronologically earlier is inferred from the ranges
    # themselves, purely to label logging and the output. Everything
    # downstream of this -- search_historical, combine_historical_tokens
    # [_by_year], serialise_field -- never branches on this label; it
    # just operates on whichever source_vectors/historical_index it's
    # handed.
    direction = "backcast" if compare_end < source_start else "forecast"

    logger.info(
        f"[dss] direction={direction} "
        f"source={source_start}-{source_end} "
        f"compare={compare_start}-{compare_end}"
    )

    source_events = load_field_events(
        lookup, q, source_start, source_end, conn=conn,
    )

    event_ids = [
        int(row[0])
        for row in source_events
    ]

    exclude_ids = set(event_ids)

    logger.info( f"[dss] source observations={len(event_ids)}" )

    if not event_ids:
        raise ValueError(
            f"no source observations for q={q!r} in "
            f"{source_start}-{source_end} -- nothing to compare. "
            f"See the [dss] phrase search log line above for why "
            f"(0 corpus occurrences vs. 0 Tier 1 matches vs. all outside "
            f"the year window are different problems)."
        )

    positions = [
        lookup.get_pos(eid)
        for eid in event_ids
    ]

    source_vectors = {
        "local": lookup.emb_local[positions],
        "medium": lookup.emb_medium[positions],
        "broad": lookup.emb_broad[positions],
    }

    trajectory = {}

    for start, end in comparison_slices(
        compare_start,
        compare_end,
        slice_width,
    ):
        logger.info( f"[dss] comparing={start}-{end}" )

        if faiss_index is not None:
            historical_index = slice_preloaded_index( faiss_index, start, end, )

            if not historical_index:
                logger.info( f"[dss] skipping empty slice={start}-{end}" )
                continue
        else:
            try:
                historical_index = load_indexes( start, end, )
            except RuntimeError:
                logger.info( f"[dss] skipping empty slice={start}-{end}" )
                continue

        matches = search_historical(
            historical_index,
            source_vectors,
            neighbours,
            exclude_ids,
        )

        if granularity == "year":
            year_results = combine_historical_tokens_by_year( matches, lookup, )

            # Sorted so trajectory stays chronological across slices too --
            # dict insertion order is what print_trajectory (and anything
            # else iterating the JSON) relies on for display order.
            for year in sorted(year_results):
                field, stats = year_results[year]
                trajectory[str(year)] = serialise_field( field, stats, period_width=1, )
        else:
            field, stats = combine_historical_tokens( matches, lookup, )
            trajectory[f"{start}-{end}"] = serialise_field( field, stats, end - start + 1, )

    return {
        "q": q,
        "direction": direction,
        "source": f"{source_start}-{source_end}",
        "compare": f"{compare_start}-{compare_end}",
        "trajectory": trajectory,
    }


def run(
    q="REVOLUTION",
    *,
    source_start,
    source_end,
    compare_start,
    compare_end,
    neighbours=10,
    slice_width=10,
    granularity="year",
    output=None,
    table_limit=5,
    lookup=None,
    conn=None,
    faiss_index=None,
):
    """
    One-shot CLI/notebook entry point: build (or reuse) resources, call
    core(), write the result to `output`, optionally print a table, and
    return the result dict. main() is a thin wrapper around this: it
    parses sys.argv and calls this with the parsed values. Calling run()
    directly (e.g. from a notebook) skips the CLI layer entirely.

    `lookup` and `faiss_index` can be passed in to reuse resources across
    repeated calls (e.g. in a notebook session) instead of rebuilding the
    FAISS index and re-opening the Zarr store every call, which is the
    dominant cost of a run. If omitted, they're built fresh here -- this
    is still the one-shot path, not the persistent-service path; for
    that, build the resources once yourself and call service() (or
    core() directly) repeatedly instead of run().

    `conn` behaves the same way: pass one in to reuse it, omit it to have
    one opened and closed per call.

    ValueError (from core(), for overlapping periods or zero source
    observations) propagates to the caller uncaught -- there's no parser
    here to hand it to. main() is the one place that catches it and
    turns it into a parser.error() exit.
    """
    owns_lookup = lookup is None

    if owns_lookup:
        source_index = CorpusFaissIndex.load_all( workers=8, )
        lookup = ZarrEventLookup( ZARR_PATH )
        lookup.attach_index( source_index )

    result = core(
        q,
        source_start=source_start,
        source_end=source_end,
        compare_start=compare_start,
        compare_end=compare_end,
        lookup=lookup,
        neighbours=neighbours,
        slice_width=slice_width,
        granularity=granularity,
        conn=conn,
        faiss_index=faiss_index,
    )

    if output is None:
        q_slug = "".join(
            c if c.isalnum() else "_"
            for c in q.strip().lower()
        )
        output = TMP_DIR / f"dss_semantic_trajectory_{q_slug}.json"

    with open( output, "w", encoding="utf8", ) as f:
        json.dump( result, f, indent=2, )

    logger.info( f"[dss] wrote {output}" )

    if table_limit > 0:
        print_trajectory(
            q,
            result["source"],
            result["direction"],
            result["trajectory"],
            limit=table_limit,
        )

    result["output"] = output
    result["lookup"] = lookup

    return result


def service(
    q,
    *,
    source_start,
    source_end,
    compare_start,
    compare_end,
    conn,
    lookup,
    faiss_index,
    neighbours=10,
    slice_width=10,
    granularity="year",
):
    """
    Single-request entry point for a persistent process (an HTTP handler,
    a WebSocket handler, a CGI-style long-lived loop -- service() has no
    opinion on protocol and doesn't do any transport itself; it just
    takes one request's worth of arguments and returns a plain result
    dict for the caller to serialise however it likes).

    Unlike run(), service() never builds or tears down resources: `conn`
    (a live DB connection), `lookup` (a ZarrEventLookup with a
    full-corpus index already attached), and `faiss_index` (a
    CorpusFaissIndex instance that has loaded every year, not just a
    range) are all required and are expected to be built once by the
    calling process at startup and passed into every service() call for
    the process's lifetime. That's what makes it safe to call this once
    per incoming request without re-opening a DB connection, re-opening
    the Zarr store, or re-reading FAISS indices off disk each time --
    the same work run() does fresh on every call, done once instead.

    Doesn't write an output file and doesn't print a table -- both are
    display/persistence concerns for a one-shot run, not a per-request
    service call. The caller decides what to do with the returned dict
    (serialise to JSON for an HTTP response, etc.).

    Raises ValueError on bad input (overlapping periods, zero source
    observations), same as core() -- propagates uncaught. The caller's
    protocol layer should catch this and translate it into whatever
    error response its transport expects (e.g. HTTP 400), the same way
    main() translates it into a parser.error() exit for the CLI.
    """
    return core(
        q,
        source_start=source_start,
        source_end=source_end,
        compare_start=compare_start,
        compare_end=compare_end,
        lookup=lookup,
        neighbours=neighbours,
        slice_width=slice_width,
        granularity=granularity,
        conn=conn,
        faiss_index=faiss_index,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--q", default="REVOLUTION", )
    parser.add_argument( "--neighbours", type=int, default=10, )
    parser.add_argument( "--source_start", type=int, default=CORPUS_MAX_YEAR - 20,
        help="Start year of the period phrase occurrences are drawn from "
             "-- these are the events whose vectors get searched against "
             "the comparison period. Defaults to a recent window, i.e. "
             "backcasting; pass an early range here (with a later "
             "--compare_start/--compare_end) to forecast instead.",
    )
    parser.add_argument( "--source_end", type=int, default=CORPUS_MAX_YEAR, )
    parser.add_argument( "--compare_start", type=int, default=1630,
        help="Start year of the period whose FAISS indices are searched "
             "for neighbours of the source events.",
    )
    parser.add_argument( "--compare_end", type=int, default=1650, )
    parser.add_argument( "--slice_width", type=int, default=10, )
    parser.add_argument( "--granularity", choices=("year", "slice"), default="year",
        help="'year' (default) reports one row per individual publication "
             "year of the matched comparison-period events. 'slice' rolls "
             "results up into --slice_width-year bands instead (the old "
             "behaviour). --slice_width still controls how many per-year "
             "FAISS indices are loaded together either way -- it's a "
             "loading batch size, not the reporting granularity, unless "
             "--granularity=slice.",
    )
    # No filesystem-safe default here -- it depends on --q, which
    # isn't known until after parsing. Resolved just below, inside run(),
    # since it's needed there too (a notebook caller of run() gets the
    # same default-output-path behaviour, not just the CLI). Passing
    # --output explicitly always overrides this.
    parser.add_argument( "--output", default=None, )
    parser.add_argument( "--table_limit", type=int, default=5, help="Top-N tokens per period to print to the terminal after the run. Set to 0 to skip printing the table.", )
    args = parser.parse_args()

    try:
        run(
            args.q,
            source_start=args.source_start,
            source_end=args.source_end,
            compare_start=args.compare_start,
            compare_end=args.compare_end,
            neighbours=args.neighbours,
            slice_width=args.slice_width,
            granularity=args.granularity,
            output=args.output,
            table_limit=args.table_limit,
        )
    except ValueError as e:
        # Re-raised through parser.error() so CLI behaviour (print usage,
        # exit non-zero) is unchanged from before the run()/main() split.
        parser.error(str(e))


if __name__ == "__main__":
    main()
