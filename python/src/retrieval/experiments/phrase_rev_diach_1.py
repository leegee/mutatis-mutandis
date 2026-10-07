#!/usr/bin/env python

"""
phrase_rev_diach_1.py

Use source-period corpus observations as semantic seeds and search a
target period with an averaged centroid, the individual seed vectors,
or both, then pool the results into one de-duplicated ranking.

For each configured phrase probe:

1. Encode the probe's target word in carrier sentences (same kind of
   vector as the stored ones: a contextual hidden state of one word).
2. Search the source period, keeping only hits whose own token is a
   seed form.
3. Pick diverse seeds (nearest first, capped per document, no two
   seeds within a few tokens of each other).
4. Reconstruct their stored MacBERTh vectors.
5. Search the target period with the centroid and/or each seed.
6. Fuse result lists by reciprocal rank, de-duplicate by event_id,
   and collapse hits that fall within a few tokens of each other in
   the same document.
7. Preserve the target token rather than requiring it to be one of
   the source seed forms, so vocabulary change can be discovered.

No database data is modified.
"""

from __future__ import annotations

import argparse
import html
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lib import corpus_config as config
from lib.corpus_config import CONCEPT_SETS, LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import load_macberth
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.models import SearchSpace


SCALE = "local"

PHRASE_PROBES = {
    "WHITE": {
        "white hair": {
            "target": "white",
            "accept": ["white", "whyte", "whit", "wyte"],
            "frames": [
                "his hair was white as snow .",
                "and the heed of hym and his heeris weren white as wolle .",
                "she had long white hair and a gentle face .",
            ],
        },
    },
}

SOURCE_YEARS = (1100, 1199)
TARGET_YEARS = (1300, 1599)

SEED_COUNT = 5
SOURCE_CANDIDATES_PER_BUCKET = 50
SEED_MAX_PER_DOC = 2
SEED_DEDUP_WINDOW = 50

TARGET_RESULTS_PER_BUCKET = 20
PASSAGE_WINDOW = 20
FINAL_N = 25
RRF_K = 60
CONTEXT_TOKENS = 80

DEFAULT_OUTPUT = config.OUT_DIR / Path("source_to_target_period.html")

# Comparing these modes helps assess whether the centroid adds information
# beyond the individual source observations.
RETRIEVAL_MODE = "both"  # "centroid", "seeds", or "both"

STOP = {
    "the", "a", "an", "and", "of", "to", "in", "that", "is", "it", "with",
    "for", "as", "his", "her", "he", "she", ",", ".", ";", ":",
}


def _norm(token: str) -> str:
    # Keep this identical to the normalisation used when indexing events.
    return unicodedata.normalize("NFKC", token).strip().lower()


DEFAULT_ACCEPT = (
    {_norm(f) for rule in CONCEPT_SETS.values() for f in rule["forms"]}
    - {_norm(f) for rule in CONCEPT_SETS.values() for f in rule["false_positives"]}
)


def normalise_vector(vector: np.ndarray) -> np.ndarray:
    """Return a finite unit vector or fail explicitly."""
    vector = np.asarray(vector, dtype=np.float32)

    if not np.all(np.isfinite(vector)):
        raise ValueError("Vector contains non-finite values.")

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        raise ValueError("Cannot normalise a near-zero vector.")

    return (vector / norm).astype(np.float32)


@dataclass
class Seed:
    probe_group: str
    phrase: str
    bucket: tuple[int, int]
    event_id: int
    distance: float
    source_rank: int | None = None

    corpus: str | None = None
    doc_id: str | None = None
    token_idx: int | None = None
    token: str | None = None
    pub_year: int | None = None
    title: str | None = None
    author: str | None = None

    vector: np.ndarray | None = None
    context: str | None = None
    centroid_similarity: float | None = None


@dataclass(frozen=True)
class QuerySpec:
    """Identity and provenance of one target retrieval query."""
    query_id: str
    kind: str
    seed_number: int | None = None


@dataclass
class TargetResult:
    """One raw hit from one target query."""
    query: QuerySpec
    bucket: tuple[int, int]
    event_id: int
    distance: float
    rank: int


@dataclass
class PooledResult:
    """One de-duplicated target observation, fused across queries."""
    event_id: int
    bucket: tuple[int, int]
    distances: dict[str, float] = field(default_factory=dict)
    mode_rrf: dict[str, float] = field(default_factory=dict)
    query_ranks: dict[str, int] = field(default_factory=dict)
    merged_event_ids: list[int] = field(default_factory=list)

    corpus: str | None = None
    doc_id: str | None = None
    token_idx: int | None = None
    token: str | None = None
    pub_year: int | None = None
    title: str | None = None
    author: str | None = None
    context: str | None = None
    lexical_accept: bool | None = None

    @property
    def score(self) -> float:
        return sum(self.mode_rrf.values())

    @property
    def best_distance(self) -> float:
        return min(self.distances.values())

    @property
    def query_ids(self) -> list[str]:
        return list(self.distances)

    def absorb(self, other: "PooledResult") -> None:
        """Keep the best contribution from each query in a passage cluster."""
        for query_id, value in other.mode_rrf.items():
            self.mode_rrf[query_id] = max(
                self.mode_rrf.get(query_id, 0.0),
                value,
            )

        for query_id, distance in other.distances.items():
            self.distances[query_id] = min(
                self.distances.get(query_id, distance),
                distance,
            )

        for query_id, rank in other.query_ranks.items():
            existing = self.query_ranks.get(query_id)
            if existing is None or rank < existing:
                self.query_ranks[query_id] = rank

        self.merged_event_ids.append(other.event_id)


def encode_target(mac, words: list[str], target: int) -> np.ndarray:
    enc = mac.tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    word_ids = enc.word_ids()

    if target not in word_ids:
        raise ValueError(
            f"Target word index {target} is not represented by the tokenizer."
        )

    pos = word_ids.index(target)

    with torch.inference_mode():
        out = mac.encode(
            input_ids=enc["input_ids"].to(mac.device),
            attention_mask=enc["attention_mask"].to(mac.device),
            return_dict=True,
        )

    vector = out.last_hidden_state[0, pos].cpu().numpy().astype(np.float32)
    return normalise_vector(vector)


def encode_probe(mac, spec: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the mean probe vector and the individual carrier vectors."""
    vectors = []

    for frame in spec["frames"]:
        words = frame.lower().split()

        if spec["target"] not in words:
            raise ValueError(
                f"target {spec['target']!r} not in frame {frame!r}"
            )

        vectors.append(
            encode_target(
                mac,
                words,
                words.index(spec["target"]),
            )
        )

    mean = np.mean(np.stack(vectors), axis=0)
    return normalise_vector(mean), vectors


def search_phrase(
    store: LanceObservationIndexStore,
    query_vector: np.ndarray,
    search_space: SearchSpace,
    top_n: int,
):
    results = []

    for bucket, results_by_scale in store.diachronic_search(
        {SCALE: query_vector},
        search_space,
        k=top_n,
    ):
        search_result = results_by_scale[SCALE]

        for rank, (event_id, distance) in enumerate(
            zip(
                search_result.event_ids,
                search_result.distances,
            ),
            start=1,
        ):
            results.append(
                (
                    bucket,
                    int(event_id),
                    float(distance),
                    rank,
                )
            )

    return results


def select_seed_candidates(results, count: int) -> list[Seed]:
    """Keep the nearest unique events as candidates for diversification."""
    best_by_event: dict[int, Seed] = {}

    for bucket, event_id, distance, rank in results:
        existing = best_by_event.get(event_id)

        if existing is None or distance < existing.distance:
            best_by_event[event_id] = Seed(
                probe_group="",
                phrase="",
                bucket=bucket,
                event_id=event_id,
                distance=distance,
                source_rank=rank,
            )

    candidates = sorted(
        best_by_event.values(),
        key=lambda s: s.distance,
    )

    return candidates[:count]


def diversify_seeds(
    candidates: list[Seed],
    count: int,
    window: int = SEED_DEDUP_WINDOW,
    max_per_doc: int = SEED_MAX_PER_DOC,
) -> list[Seed]:
    """Select nearest candidates while preventing passage domination."""
    chosen: list[Seed] = []
    per_doc: dict[tuple, int] = defaultdict(int)

    for cand in sorted(candidates, key=lambda s: s.distance):
        if cand.doc_id is None or cand.token_idx is None:
            continue

        key = (cand.corpus, cand.doc_id)

        if per_doc[key] >= max_per_doc:
            continue

        if any(
            (s.corpus, s.doc_id) == key
            and abs(s.token_idx - cand.token_idx) <= window
            for s in chosen
        ):
            continue

        chosen.append(cand)
        per_doc[key] += 1

        if len(chosen) == count:
            break

    return chosen


def fetch_tokens(connection, event_ids) -> dict[int, str]:
    """Map event_id -> normalised token."""
    ids = sorted(set(event_ids))

    if not ids:
        return {}

    with connection.cursor() as cur:
        cur.execute(
            "SELECT event_id, token FROM events WHERE event_id = ANY(%s)",
            (ids,),
        )
        return {
            int(row[0]): _norm(row[1])
            for row in cur.fetchall()
        }


def fetch_event_metadata(connection, items) -> None:
    """Populate corpus/document/token metadata on event-bearing objects."""
    if not items:
        return

    event_ids = sorted({item.event_id for item in items})

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT e.event_id, e.corpus, e.doc_id, e.token_idx,
                   e.token, e.pub_year, d.title, d.author
            FROM events e
            JOIN documents d
              ON d.corpus = e.corpus AND d.doc_id = e.doc_id
            WHERE e.event_id = ANY(%s)
            """,
            (event_ids,),
        )
        by_event = {
            int(row[0]): row
            for row in cursor.fetchall()
        }

    for item in items:
        row = by_event.get(item.event_id)

        if row is None:
            logger.warning(
                "No PostgreSQL metadata for event_id=%s",
                item.event_id,
            )
            continue

        (
            _,
            item.corpus,
            item.doc_id,
            item.token_idx,
            item.token,
            item.pub_year,
            item.title,
            item.author,
        ) = row


def fetch_context(
    connection,
    corpus: str | None,
    doc_id: str | None,
    token_idx: int | None,
    radius: int,
) -> str:
    """Return HTML-escaped token context from PostgreSQL."""
    if corpus is None or doc_id is None or token_idx is None:
        return ""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT token
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
        rows = cursor.fetchall()

    return " ".join(
        html.escape(str(row[0]))
        for row in rows
    )


def reconstruct_seed_vectors(
    store: LanceObservationIndexStore,
    seeds: list[Seed],
) -> int:
    """Reconstruct stored seed vectors from their source Lance tables."""
    by_bucket = defaultdict(list)

    for seed in seeds:
        by_bucket[seed.bucket].append(seed)

    missing = 0

    for bucket, bucket_seeds in by_bucket.items():
        event_ids = [seed.event_id for seed in bucket_seeds]

        indexes = store.get(
            SearchSpace(
                years=bucket,
                scale=(SCALE,),
            )
        )

        vectors = np.asarray(
            indexes[SCALE].reconstruct_many(event_ids)
        )

        if vectors.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D vectors for {bucket}, "
                f"got shape {vectors.shape}"
            )

        if len(vectors) != len(bucket_seeds):
            raise RuntimeError(
                f"Vector count mismatch for {bucket}: "
                f"{len(vectors)} vectors for "
                f"{len(bucket_seeds)} seeds"
            )

        for seed, vector in zip(bucket_seeds, vectors):
            try:
                seed.vector = normalise_vector(vector)
            except ValueError:
                missing += 1

    return missing


def make_centroid(seeds: list[Seed]) -> np.ndarray:
    """Average normalised seed vectors and renormalise."""
    vectors = [
        seed.vector
        for seed in seeds
        if seed.vector is not None
    ]

    if not vectors:
        raise RuntimeError(
            "Cannot construct centroid: no valid seed vectors."
        )

    centroid = normalise_vector(
        np.mean(np.stack(vectors), axis=0)
    )

    for seed in seeds:
        if seed.vector is not None:
            seed.centroid_similarity = float(
                np.dot(seed.vector, centroid)
            )

    return centroid


def search_vector(
    store: LanceObservationIndexStore,
    vector: np.ndarray,
    search_space: SearchSpace,
    top_n: int,
):
    """Search each target chronological bucket independently."""
    results = []

    for bucket, results_by_scale in store.diachronic_search(
        {SCALE: vector},
        search_space,
        k=top_n,
    ):
        search_result = results_by_scale[SCALE]

        for rank, (event_id, distance) in enumerate(
            zip(
                search_result.event_ids,
                search_result.distances,
            ),
            start=1,
        ):
            results.append(
                (
                    bucket,
                    int(event_id),
                    float(distance),
                    rank,
                )
            )

    return results


def make_target_results(
    store: LanceObservationIndexStore,
    seeds: list[Seed],
    centroid: np.ndarray,
    search_space: SearchSpace,
    top_n: int,
    retrieval_mode: str,
) -> list[TargetResult]:
    """Run the configured centroid and/or individual seed queries."""
    results = []

    if retrieval_mode in {"centroid", "both"}:
        query = QuerySpec(
            query_id="centroid",
            kind="centroid",
        )

        for bucket, event_id, distance, rank in search_vector(
            store,
            centroid,
            search_space,
            top_n,
        ):
            results.append(
                TargetResult(
                    query=query,
                    bucket=bucket,
                    event_id=event_id,
                    distance=distance,
                    rank=rank,
                )
            )

    if retrieval_mode in {"seeds", "both"}:
        for seed_number, seed in enumerate(seeds, start=1):
            if seed.vector is None:
                continue

            query = QuerySpec(
                query_id=f"seed-{seed_number}",
                kind="seed",
                seed_number=seed_number,
            )

            logger.info(
                "target search: seed %d/%d — %s",
                seed_number,
                len(seeds),
                seed.phrase,
            )

            for bucket, event_id, distance, rank in search_vector(
                store,
                seed.vector,
                search_space,
                top_n,
            ):
                results.append(
                    TargetResult(
                        query=query,
                        bucket=bucket,
                        event_id=event_id,
                        distance=distance,
                        rank=rank,
                    )
                )

    return results


def pool_results(
    results: list[TargetResult],
    rrf_k: int = RRF_K,
) -> list[PooledResult]:
    """Fuse query rankings while preserving within-bucket rank provenance."""
    by_query_bucket: dict[
        tuple[str, tuple[int, int]],
        dict[int, TargetResult],
    ] = defaultdict(dict)

    for result in results:
        key = (result.query.query_id, result.bucket)
        existing = by_query_bucket[key].get(result.event_id)

        if existing is None or result.distance < existing.distance:
            by_query_bucket[key][result.event_id] = result

    pooled: dict[int, PooledResult] = {}

    for (query_id, bucket), hits in by_query_bucket.items():
        for hit in hits.values():
            p = pooled.setdefault(
                hit.event_id,
                PooledResult(
                    event_id=hit.event_id,
                    bucket=hit.bucket,
                ),
            )

            contribution = 1.0 / (rrf_k + hit.rank)

            p.mode_rrf[query_id] = max(
                p.mode_rrf.get(query_id, 0.0),
                contribution,
            )

            p.distances[query_id] = min(
                p.distances.get(query_id, hit.distance),
                hit.distance,
            )

            p.query_ranks[query_id] = min(
                p.query_ranks.get(query_id, hit.rank),
                hit.rank,
            )

    return sorted(
        pooled.values(),
        key=lambda p: (-p.score, p.best_distance),
    )


def collapse_passages(
    pooled: list[PooledResult],
    window: int = PASSAGE_WINDOW,
) -> list[PooledResult]:
    """Fold nearby observations into the best-ranked passage anchor."""
    kept: list[PooledResult] = []
    anchors_by_doc: dict[
        tuple,
        list[PooledResult],
    ] = defaultdict(list)

    for item in pooled:
        if item.doc_id is None or item.token_idx is None:
            kept.append(item)
            continue

        # doc_id is part of the identity because titles are not unique
        # across an EEBO corpus and must never determine passage identity.
        anchors = anchors_by_doc[
            (item.corpus, item.doc_id)
        ]

        for anchor in anchors:
            if abs(anchor.token_idx - item.token_idx) <= window:
                anchor.absorb(item)
                break
        else:
            anchors.append(item)
            kept.append(item)

    kept.sort(
        key=lambda p: (-p.score, p.best_distance)
    )

    return kept


def log_seed_coherence(seeds: list[Seed]) -> None:
    similarities = [
        seed.centroid_similarity
        for seed in seeds
        if seed.centroid_similarity is not None
    ]

    if not similarities:
        return

    logger.info(
        "seed-centroid coherence: mean=%.4f min=%.4f max=%.4f",
        np.mean(similarities),
        np.min(similarities),
        np.max(similarities),
    )


def print_seed_summary(
    probe_group: str,
    phrase: str,
    seeds: list[Seed],
) -> None:
    logger.info(" ")
    logger.info("%s / %s", probe_group, phrase)

    for number, seed in enumerate(seeds, start=1):
        coherence = (
            f" centroid={seed.centroid_similarity:.4f}"
            if seed.centroid_similarity is not None
            else ""
        )

        logger.info(
            "  seed %d: %s %s — %s — "
            "doc_id=%s source-rank=%s d=%.4f%s",
            number,
            seed.pub_year or "?",
            seed.token or "?",
            seed.title or "(untitled)",
            seed.doc_id or "?",
            seed.source_rank or "?",
            seed.distance,
            coherence,
        )


CSS = """
html, body { background: #111; color: #ddd; font-family: system-ui, sans-serif; margin: 0; padding: 0; }
body { padding: 24px; }
h1 { font-size: 22px; font-weight: 500; }
h2 { font-size: 18px; font-weight: 500; margin-top: 40px; }
h3 { font-size: 16px; font-weight: 500; margin-top: 28px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
th, td { border-bottom: 1px solid #333; padding: 8px; text-align: left; vertical-align: top; }
th { color: #aaa; font-weight: 500; }
.context { max-width: 900px; line-height: 1.5; }
.seed { border: 1px solid #333; padding: 14px; margin: 12px 0; }
.meta { color: #999; font-size: 13px; }
.doc-id { color: #aaa; font-family: monospace; font-size: 12px; }
.lexical-no { color: #e6a15a; }
hr { border: 0; border-top: 1px solid #333; margin: 40px 0; }
"""


def format_query(
    query_id: str,
    distances: dict[str, float],
    ranks: dict[str, int],
    seeds: list[Seed],
) -> str:
    if query_id == "centroid":
        return (
            f"centroid "
            f"(r{ranks.get(query_id, '?')}, "
            f"d={distances[query_id]:.3f})"
        )

    if query_id.startswith("seed-"):
        try:
            number = int(query_id.split("-", 1)[1])
            seed = seeds[number - 1]
            label = (
                f"seed-{number}: "
                f"{seed.token or '?'}"
            )
        except (ValueError, IndexError):
            label = query_id
    else:
        label = query_id

    return (
        f"{label} "
        f"(r{ranks.get(query_id, '?')}, "
        f"d={distances[query_id]:.3f})"
    )


def write_html(
    output_path: Path,
    seed_data,
    pooled_by_phrase,
    retrieval_mode: str,
):
    """Write seeds, diagnostics and pooled target passages."""
    chunks = [
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>MacBERTh source-to-target search</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>MacBERTh source-to-target search</h1>\n"
        f'<p class="meta">Retrieval mode: '
        f"{html.escape(retrieval_mode)}</p>\n"
    ]

    for probe_group, phrase, seeds, probe_diagnostics in seed_data:
        pooled = pooled_by_phrase[(probe_group, phrase)]

        if retrieval_mode == "both":
            n_queries = len(seeds) + 1
        elif retrieval_mode == "centroid":
            n_queries = 1
        else:
            n_queries = len(seeds)

        chunks.append(
            f"<h2>{html.escape(probe_group)}: "
            f"{html.escape(phrase)}</h2>"
        )

        chunks.append("<h3>Probe diagnostics</h3>")

        chunks.append(
            f'<p class="meta">'
            f"Carrier frames: {probe_diagnostics['frame_count']} · "
            f"seed-centroid coherence: "
            f"mean={probe_diagnostics['coherence_mean']:.4f}, "
            f"min={probe_diagnostics['coherence_min']:.4f}, "
            f"max={probe_diagnostics['coherence_max']:.4f}"
            f"</p>"
        )

        chunks.append("<h3>Source-period seeds</h3>")

        for number, seed in enumerate(seeds, start=1):
            chunks.append(
                f"""
<div class="seed">
<b>Seed {number}</b>
<div class="meta">
{seed.pub_year or '?'} · event_id={seed.event_id}
· source rank={seed.source_rank or '?'}
· distance={seed.distance:.4f}
· centroid similarity={seed.centroid_similarity:.4f}
</div>
<p>
{html.escape(seed.author or "(unknown author)")}
— {html.escape(seed.title or "(untitled)")}
<br>
<span class="doc-id">
doc_id={html.escape(seed.doc_id or "?")}
</span>
</p>
<p class="context">{seed.context or ""}</p>
</div>
"""
            )

        chunks.append("<h3>Pooled target retrievals</h3>")

        chunks.append(
            '<p class="meta">'
            "Ranked by reciprocal-rank fusion using within-bucket "
            "query ranks. Target tokens are not lexically filtered: "
            "the lexical column is diagnostic only. Nearby observations "
            "in the same document are collapsed into passage clusters. "
            "Document identity is given by doc_id; titles are not assumed "
            "to be unique."
            "</p>"
        )

        chunks.append(
            "<table>"
            "<tr>"
            "<th>#</th>"
            "<th>Year</th>"
            "<th>Token</th>"
            "<th>Found by</th>"
            "<th>Best d</th>"
            "<th>Lexical</th>"
            "<th>Author</th>"
            "<th>Title / Doc ID</th>"
            "<th>Context</th>"
            "</tr>"
        )

        for rank, result in enumerate(pooled, start=1):
            found = ", ".join(
                format_query(
                    query_id,
                    result.distances,
                    result.query_ranks,
                    seeds,
                )
                for query_id in result.query_ids
            )

            merged = (
                f" +{len(result.merged_event_ids)} merged"
                if result.merged_event_ids
                else ""
            )

            lexical = (
                "yes"
                if result.lexical_accept
                else "no"
                if result.lexical_accept is False
                else "?"
            )

            lexical_class = (
                ""
                if result.lexical_accept
                else ' class="lexical-no"'
                if result.lexical_accept is False
                else ""
            )

            chunks.append(
                f"<tr>"
                f"<td>{rank}</td>"
                f"<td>{result.pub_year or '?'}</td>"
                f"<td>{html.escape(result.token or '?')}"
                f"<br><span class=\"meta\">"
                f"idx {result.token_idx}{merged}"
                f"</span></td>"
                f"<td>{len(result.query_ids)}/{n_queries}"
                f"<br><span class=\"meta\">"
                f"{html.escape(found)}"
                f"</span></td>"
                f"<td>{result.best_distance:.4f}</td>"
                f"<td{lexical_class}>{lexical}</td>"
                f"<td>{html.escape(result.author or '(unknown)')}</td>"
                f"<td>"
                f"{html.escape(result.title or '(untitled)')}"
                f"<br>"
                f"<span class=\"doc-id\">"
                f"doc_id={html.escape(result.doc_id or '?')}"
                f"</span>"
                f"</td>"
                f'<td class="context">{result.context or ""}</td>'
                f"</tr>"
            )

        chunks.append("</table><hr>")

    chunks.append("</body>\n</html>\n")

    output_path.write_text(
        "".join(chunks),
        encoding="utf-8",
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=(
            "Use source-period MacBERTh observations as semantic seeds "
            "for pooled searches in a target period."
        )
    )

    parser.add_argument(
        "--seed-count",
        type=int,
        default=SEED_COUNT,
        help=f"Seeds used per phrase (default: {SEED_COUNT}).",
    )

    parser.add_argument(
        "--seed-pool",
        type=int,
        default=SOURCE_CANDIDATES_PER_BUCKET,
        help=(
            "Candidates retrieved per source bucket before filtering "
            "and diversification "
            f"(default: {SOURCE_CANDIDATES_PER_BUCKET})."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TARGET_RESULTS_PER_BUCKET,
        help=(
            "Results per chronological bucket per target query "
            f"(default: {TARGET_RESULTS_PER_BUCKET})."
        ),
    )

    parser.add_argument(
        "--final-n",
        type=int,
        default=FINAL_N,
        help=f"Pooled passages kept per phrase (default: {FINAL_N}).",
    )

    parser.add_argument(
        "--passage-window",
        type=int,
        default=PASSAGE_WINDOW,
        help=(
            "Tokens within which hits in one document are merged "
            f"(default: {PASSAGE_WINDOW})."
        ),
    )

    parser.add_argument(
        "--context",
        type=int,
        default=CONTEXT_TOKENS,
        help="Tokens of context either side of a hit.",
    )

    parser.add_argument(
        "--mode",
        choices=("centroid", "seeds", "both"),
        default=RETRIEVAL_MODE,
        help=(
            "Target retrieval strategy: centroid only, seeds only, "
            f"or both (default: {RETRIEVAL_MODE})."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML file.",
    )

    args = parser.parse_args()

    if args.seed_count < 1:
        parser.error("--seed-count must be at least 1")

    if args.seed_pool < args.seed_count:
        parser.error(
            "--seed-pool must be at least --seed-count"
        )

    if args.top_n < 1:
        parser.error("--top-n must be at least 1")

    if args.final_n < 1:
        parser.error("--final-n must be at least 1")

    if args.passage_window < 0:
        parser.error(
            "--passage-window must not be negative"
        )

    if args.context < 0:
        parser.error("--context must not be negative")

    source_years = SOURCE_YEARS
    target_years = TARGET_YEARS

    logger.info(
        "source seed period: %d–%d",
        *source_years,
    )
    logger.info(
        "target period:      %d–%d",
        *target_years,
    )
    logger.info(
        "seed count:          %d (pool %d/bucket)",
        args.seed_count,
        args.seed_pool,
    )
    logger.info(
        "target top-N/bucket: %d",
        args.top_n,
    )
    logger.info(
        "retrieval mode:      %s",
        args.mode,
    )
    logger.info(
        "scale:               %s",
        SCALE,
    )
    logger.info("Loading MacBERTh...")

    mac = load_macberth()
    store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR
    )

    source_space = SearchSpace(
        years=source_years,
        scale=(SCALE,),
    )

    target_space = SearchSpace(
        years=target_years,
        scale=(SCALE,),
    )

    connection = get_connection()

    seed_data = []
    pooled_by_phrase: dict[
        tuple[str, str],
        list[PooledResult],
    ] = {}

    for probe_group, probes in PHRASE_PROBES.items():
        for phrase, spec in probes.items():
            accept = {
                _norm(form)
                for form in spec.get(
                    "accept",
                    DEFAULT_ACCEPT,
                )
            }

            logger.info(" ")
            logger.info(
                "source search %s: %r",
                probe_group,
                phrase,
            )

            query_vector, frame_vectors = encode_probe(
                mac,
                spec,
            )

            source_results = search_phrase(
                store,
                query_vector,
                source_space,
                args.seed_pool,
            )

            src_tokens = fetch_tokens(
                connection,
                [result[1] for result in source_results],
            )

            before = len(source_results)

            source_results = [
                result
                for result in source_results
                if src_tokens.get(result[1]) in accept
            ]

            logger.info(
                "source hits: %d -> %d after token filter",
                before,
                len(source_results),
            )

            candidates = select_seed_candidates(
                source_results,
                args.seed_pool,
            )

            for candidate in candidates:
                candidate.probe_group = probe_group
                candidate.phrase = phrase

            fetch_event_metadata(
                connection,
                candidates,
            )

            seeds = diversify_seeds(
                candidates,
                args.seed_count,
            )

            for seed in seeds:
                seed.context = fetch_context(
                    connection,
                    seed.corpus,
                    seed.doc_id,
                    seed.token_idx,
                    args.context,
                )

            missing = reconstruct_seed_vectors(
                store,
                seeds,
            )

            if missing:
                logger.warning(
                    "Could not reconstruct %d seed vector(s)",
                    missing,
                )

            seeds = [
                seed
                for seed in seeds
                if seed.vector is not None
            ]

            if not seeds:
                logger.warning(
                    "No usable seed vectors for %r "
                    "(try a larger --seed-pool or check "
                    "the probe's accept list)",
                    phrase,
                )
                continue

            centroid = make_centroid(seeds)

            log_seed_coherence(seeds)
            print_seed_summary(
                probe_group,
                phrase,
                seeds,
            )

            logger.info(
                "target searches: %r",
                phrase,
            )

            raw = make_target_results(
                store,
                seeds,
                centroid,
                target_space,
                args.top_n,
                args.mode,
            )

            target_tokens = fetch_tokens(
                connection,
                [result.event_id for result in raw],
            )

            # Target lexical identity is diagnostic only. Keeping these
            # observations is what allows semantic drift to be discovered.
            counts = Counter(
                token
                for token in target_tokens.values()
                if token not in STOP
            )

            logger.info(
                "target content-token distribution for %r: %s",
                phrase,
                counts.most_common(40),
            )

            pooled = pool_results(raw)

            fetch_event_metadata(
                connection,
                pooled,
            )

            for result in pooled:
                token = target_tokens.get(result.event_id)
                result.lexical_accept = (
                    token in accept
                    if token is not None
                    else None
                )

            pooled = collapse_passages(
                pooled,
                args.passage_window,
            )

            pooled = pooled[: args.final_n]

            for result in pooled:
                result.context = fetch_context(
                    connection,
                    result.corpus,
                    result.doc_id,
                    result.token_idx,
                    args.context,
                )

            logger.info(
                "%r: %d raw hits -> %d pooled passages",
                phrase,
                len(raw),
                len(pooled),
            )

            pooled_by_phrase[
                (probe_group, phrase)
            ] = pooled

            similarities = [
                seed.centroid_similarity
                for seed in seeds
                if seed.centroid_similarity is not None
            ]

            probe_diagnostics = {
                "frame_count": len(frame_vectors),
                "coherence_mean": float(np.mean(similarities)),
                "coherence_min": float(np.min(similarities)),
                "coherence_max": float(np.max(similarities)),
            }

            seed_data.append(
                (
                    probe_group,
                    phrase,
                    seeds,
                    probe_diagnostics,
                )
            )

    logger.info(" ")
    logger.info(
        "writing: %s",
        args.output,
    )

    write_html(
        args.output,
        seed_data,
        pooled_by_phrase,
        args.mode,
    )

    logger.info("done")


if __name__ == "__main__":
    main()