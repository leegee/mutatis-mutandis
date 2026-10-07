#!/usr/bin/env python

"""
Cluster MacBERTh phrase-search observations and produce interactive
2-D PCA plots with hoverable source context.

Each chronological bucket is clustered independently.

The observations are the union of the top-N-per-bucket results from
the configured phrase probes.  Duplicate event IDs within a bucket
are merged, preserving all probe retrievals.

The actual Lance vectors are reconstructed and clustered using a
cosine-distance threshold.

The resulting HTML contains one interactive Plotly figure per
chronological bucket.

No database data is modified.
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html

from lib.corpus_config import LANCE_INDEXES_DIR, CORPUS_MIN_YEAR, CORPUS_MAX_YEAR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.macberth_phrase_encoder2 import MacBertMeanPhraseEncoder
from retrieval.models import SearchSpace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALE = "medium"

PHRASE_PROBES = {
    "WHITE": [
        "he had white hair",
        "his hair was white",
        "hair white as wool",
    ],
    "GREY": [
        "he had grey hair",
        "his hair was grey",
        "hair grey as wool",
    ],
}

MIN_YEAR = CORPUS_MIN_YEAR
MAX_YEAR = CORPUS_MAX_YEAR

TOP_N = 10

CLUSTER_DISTANCE = 0.080

CONTEXT_TOKENS = 80

DEFAULT_OUTPUT = Path("phrase_clusters.html")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Retrieval:
    probe_group: str
    phrase: str
    distance: float


@dataclass
class Observation:
    bucket: tuple[int, int]
    event_id: int
    retrievals: list[Retrieval] = field(default_factory=list)

    corpus: str | None = None
    doc_id: str | None = None
    token_idx: int | None = None
    token: str | None = None
    pub_year: int | None = None
    medium_window_id: int | None = None
    medium_window_token_pos: int | None = None
    title: str | None = None
    author: str | None = None

    context: str | None = None
    vector: np.ndarray | None = None

    cluster: int | None = None
    x: float | None = None
    y: float | None = None


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

def search_phrase(
    store: LanceObservationIndexStore,
    encoder: MacBertMeanPhraseEncoder,
    phrase: str,
    probe_group: str,
    search_space: SearchSpace,
    top_n: int,
):
    """
    Return top-N results from every chronological bucket.
    """

    query_vector = encoder.encode_text(phrase)

    queries_by_scale = {
        SCALE: query_vector,
    }

    results = []

    for bucket, results_by_scale in store.diachronic_search(
        queries_by_scale,
        search_space,
        k=top_n,
    ):
        search_result = results_by_scale[SCALE]

        for event_id, distance in zip(
            search_result.event_ids,
            search_result.distances,
        ):
            results.append(
                (
                    bucket,
                    int(event_id),
                    float(distance),
                    probe_group,
                    phrase,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Deduplication / merging
# ---------------------------------------------------------------------------

def merge_results(results):
    """
    Merge repeated event IDs within each chronological bucket.

    An event retrieved by several probes becomes one Observation,
    retaining all of its retrieval information.
    """

    observations = {}

    for bucket, event_id, distance, probe_group, phrase in results:
        key = (bucket, event_id)

        observation = observations.get(key)

        if observation is None:
            observation = Observation(
                bucket=bucket,
                event_id=event_id,
            )
            observations[key] = observation

        observation.retrievals.append(
            Retrieval(
                probe_group=probe_group,
                phrase=phrase,
                distance=distance,
            )
        )

    return list(observations.values())


# ---------------------------------------------------------------------------
# PostgreSQL metadata
# ---------------------------------------------------------------------------

def fetch_metadata(connection, observations):
    """
    Populate authoritative event/document metadata from PostgreSQL.
    """

    if not observations:
        return

    event_ids = sorted(
        {
            observation.event_id
            for observation in observations
        }
    )

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
                e.medium_window_id,
                e.medium_window_token_pos,
                d.title,
                d.author
            FROM events e
            JOIN documents d
                ON d.corpus = e.corpus
                AND d.doc_id = e.doc_id
            WHERE e.event_id = ANY(%s)
            """,
            (event_ids,),
        )

        rows = cursor.fetchall()

    by_event = {
        int(row[0]): row
        for row in rows
    }

    for observation in observations:
        row = by_event.get(observation.event_id)

        if row is None:
            logger.warning(
                "No PostgreSQL metadata for event_id=%s",
                observation.event_id,
            )
            continue

        (
            event_id,
            corpus,
            doc_id,
            token_idx,
            token,
            pub_year,
            medium_window_id,
            medium_window_token_pos,
            title,
            author,
        ) = row

        observation.corpus = corpus
        observation.doc_id = doc_id
        observation.token_idx = token_idx
        observation.token = token
        observation.pub_year = pub_year
        observation.medium_window_id = medium_window_id
        observation.medium_window_token_pos = medium_window_token_pos
        observation.title = title
        observation.author = author


def fetch_context(
    connection,
    observation: Observation,
    radius: int,
):
    if (
        observation.corpus is None
        or observation.doc_id is None
        or observation.token_idx is None
    ):
        return ""

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
                observation.corpus,
                observation.doc_id,
                max(0, observation.token_idx - radius),
                observation.token_idx + radius,
            ),
        )

        rows = cursor.fetchall()

    words = []

    for token_idx, token in rows:
        token = str(token)

        if token_idx == observation.token_idx:
            token = f"▶<u>{html.escape(token)}</u>◀"
        else:
            token = html.escape(token)

        words.append(token)

    return " ".join(words)


# ---------------------------------------------------------------------------
# Lance vectors
# ---------------------------------------------------------------------------

def fetch_vectors(
    store: LanceObservationIndexStore,
    observations,
):
    """
    Reconstruct the actual stored Lance vectors.

    Vectors are fetched bucket-by-bucket because the observations live
    in chronological indexes.
    """

    by_bucket = defaultdict(list)

    for observation in observations:
        by_bucket[observation.bucket].append(observation)

    missing = 0

    for bucket, bucket_observations in by_bucket.items():

        event_ids = [
            observation.event_id
            for observation in bucket_observations
        ]

        search_space = SearchSpace(
            years=bucket,
            scale=(SCALE,),
        )

        indexes = store.get(search_space)
        lance_index = indexes[SCALE]

        vectors = lance_index.reconstruct_many(event_ids)

        vectors = np.asarray(vectors)

        if vectors.ndim != 2:
            raise RuntimeError(
                f"Expected a 2-D vector array for {bucket}, "
                f"got shape {vectors.shape}"
            )

        if len(vectors) != len(bucket_observations):
            raise RuntimeError(
                f"Vector count mismatch for {bucket}: "
                f"{len(vectors)} vectors for "
                f"{len(bucket_observations)} observations"
            )

        for observation, vector in zip(
            bucket_observations,
            vectors,
        ):
            vector = np.asarray(vector, dtype=np.float32)

            if not np.all(np.isfinite(vector)):
                observation.vector = None
                missing += 1
                continue

            norm = np.linalg.norm(vector)

            if norm < 1e-12:
                observation.vector = None
                missing += 1
                continue

            observation.vector = vector / norm

    return missing


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_threshold(vectors, threshold):
    """
    Threshold clustering using a cosine-distance graph.

    Two observations are connected when:

        cosine_distance <= threshold

    Connected components form clusters.

    This is deliberately simple and transparent; it does not impose
    a particular number of clusters.
    """

    n = len(vectors)

    if n == 0:
        return []

    if n == 1:
        return [[0]]

    matrix = np.asarray(vectors, dtype=np.float32)

    # Vectors are already normalized.
    similarity = matrix @ matrix.T
    distances = 1.0 - similarity

    adjacency = distances <= threshold

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            neighbours = np.flatnonzero(
                adjacency[current]
                & ~visited
            )

            for neighbour in neighbours:
                visited[neighbour] = True
                stack.append(int(neighbour))

        clusters.append(component)

    clusters.sort(
        key=lambda cluster: (-len(cluster), min(cluster))
    )

    return clusters


def assign_clusters(observations, threshold):
    """
    Cluster observations independently within each chronological bucket.
    """

    by_bucket = defaultdict(list)

    for observation in observations:
        if observation.vector is not None:
            by_bucket[observation.bucket].append(observation)

    bucket_clusters = {}

    for bucket, bucket_observations in sorted(by_bucket.items()):

        vectors = [
            observation.vector
            for observation in bucket_observations
        ]

        clusters = cluster_threshold(
            vectors,
            threshold,
        )

        bucket_clusters[bucket] = []

        for cluster_number, indices in enumerate(clusters, start=1):

            cluster_observations = [
                bucket_observations[index]
                for index in indices
            ]

            for observation in cluster_observations:
                observation.cluster = cluster_number

            bucket_clusters[bucket].append(
                cluster_observations
            )

    return bucket_clusters


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_2d(vectors):
    """
    Minimal PCA implementation using NumPy SVD.

    This avoids introducing sklearn as another dependency.
    """

    matrix = np.asarray(vectors, dtype=np.float64)

    if len(matrix) == 0:
        return np.empty((0, 2))

    if len(matrix) == 1:
        return np.zeros((1, 2))

    centred = matrix - matrix.mean(axis=0)

    u, singular_values, _ = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    components = min(2, u.shape[1])

    result = u[:, :components] * singular_values[:components]

    if components == 1:
        result = np.column_stack(
            [result[:, 0], np.zeros(len(result))]
        )

    return result


def assign_pca_coordinates(cluster_observations):
    """
    Calculate one PCA projection per chronological bucket.
    """

    by_bucket = defaultdict(list)

    for observation in cluster_observations:
        if observation.vector is not None:
            by_bucket[observation.bucket].append(observation)

    for bucket, observations in by_bucket.items():

        coordinates = pca_2d(
            [
                observation.vector
                for observation in observations
            ]
        )

        for observation, (x, y) in zip(
            observations,
            coordinates,
        ):
            observation.x = float(x)
            observation.y = float(y)


# ---------------------------------------------------------------------------
# Hover text
# ---------------------------------------------------------------------------

def make_hover(observation: Observation):
    title = "<br>".join(
        textwrap.wrap(
            html.escape(observation.title or "(untitled)"),
            width=90,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )

    author = html.escape(
        observation.author or "(unknown author)"
    )

    token = html.escape(
        observation.token or ""
    )

    context = observation.context or ""

    context = "<br>".join(
        textwrap.wrap(
            context,
            width=90,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )

    context = context.replace(
        "\x00MARK_OPEN\x00",
        "<mark>",
    ).replace(
        "\x00MARK_CLOSE\x00",
        "</mark>",
    )

    retrieval_lines = []

    for retrieval in sorted(
        observation.retrievals,
        key=lambda item: item.distance,
    ):
        retrieval_lines.append(
            f"{html.escape(retrieval.probe_group)}: "
            f"{html.escape(retrieval.phrase)} "
            f"(d={retrieval.distance:.4f})"
        )

    retrieval_text = "<br>".join(retrieval_lines)

    return (
        f"<b>{observation.pub_year or '?'} — "
        f"{author}</b><br>"
        f"{title}<br>"
        f"event_id={observation.event_id}<br>"
        f"cluster={observation.cluster}<br>"
        f"token={token}<br>"
        f"<br>"
        f"{retrieval_text}"
        f"<br><br>"
        f"<b>Context</b><br>"
        f"{context}"
    )

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_bucket_figure(
    bucket,
    observations,
    threshold,
):
    """
    Produce one Plotly scatter plot for a chronological bucket.
    """

    grouped = defaultdict(list)

    for observation in observations:
        grouped[
            observation.cluster
        ].append(observation)

    traces = []

    for cluster_number in sorted(grouped):
        cluster_observations = grouped[cluster_number]

        traces.append(
            go.Scatter(
                x=[
                    observation.x
                    for observation in cluster_observations
                ],
                y=[
                    observation.y
                    for observation in cluster_observations
                ],
                mode="markers",
                name=(
                    f"Cluster {cluster_number} "
                    f"({len(cluster_observations)})"
                ),
                text=[
                    make_hover(observation)
                    for observation in cluster_observations
                ],
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": 9,
                    "opacity": 0.85,
                },
            )
        )

    figure = go.Figure(
        data=traces
    )

    cluster_count = len(grouped)

    figure.update_layout(
        title=(
            f"{bucket[0]}–{bucket[1]}  "
            f"·  {len(observations)} observations  "
            f"·  {cluster_count} clusters  "
            f"·  threshold {threshold:.3f}"
        ),
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font={
            "color": "#dddddd",
        },
        hoverlabel={
            "bgcolor": "#202020",
            "font": {
                "color": "#eeeeee",
                "size": 13,
            },
            "align": "left",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 60,
            "r": 30,
            "t": 100,
            "b": 60,
        },
        xaxis={
            "title": "PCA dimension 1",
            "zeroline": False,
            "gridcolor": "#292929",
        },
        yaxis={
            "title": "PCA dimension 2",
            "zeroline": False,
            "gridcolor": "#292929",
        },
    )

    return figure


def write_html(
    bucket_figures,
    output_path,
):
    """
    Write all bucket plots into one self-contained HTML document.

    Plotly.js is embedded exactly once, in the first figure.
    """

    chunks = []

    chunks.append(
        """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MacBERTh phrase clusters</title>
<style>
html, body {
    background: #111;
    color: #ddd;
    font-family: system-ui, sans-serif;
    margin: 0;
    padding: 0;
}

body {
    padding: 24px;
}

h1 {
    font-size: 22px;
    font-weight: 500;
}

.bucket {
    margin-bottom: 50px;
}

hr {
    border: 0;
    border-top: 1px solid #333;
    margin: 40px 0;
}
</style>
</head>
<body>
<h1>MacBERTh phrase clusters</h1>
"""
    )

    for index, (bucket, figure) in enumerate(bucket_figures):

        if index > 0:
            chunks.append(
                '<hr>'
            )

        chunks.append(
            '<div class="bucket">'
        )

        chunks.append(
            to_html(
                figure,
                full_html=False,
                include_plotlyjs=(
                    "inline"
                    if index == 0
                    else False
                ),
                config={
                    "responsive": True,
                    "displaylogo": False,
                },
            )
        )

        chunks.append(
            "</div>"
        )

    chunks.append(
        """
</body>
</html>
"""
    )

    output_path.write_text(
        "".join(chunks),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(bucket_clusters):
    logger.info(" ")
    logger.info("=" * 80)
    logger.info("CLUSTER SUMMARY")
    logger.info("=" * 80)

    for bucket, clusters in sorted(
        bucket_clusters.items()
    ):
        sizes = sorted(
            (
                len(cluster)
                for cluster in clusters
            ),
            reverse=True,
        )

        logger.info(
            f"{bucket[0]}–{bucket[1]} "
            f"({sum(sizes)} observations)"
        )

        logger.info(
            f"  {len(clusters)} clusters "
            f"sizes={sizes}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(
        encoding="utf-8"
    )

    parser = argparse.ArgumentParser(
        description=( "Cluster MacBERTh phrase-search observations and create interactive PCA plots." )
    )

    parser.add_argument( "--distance", type=float, default=CLUSTER_DISTANCE, help=( f"Cosine-distance clustering threshold (default: {CLUSTER_DISTANCE})" ), )

    parser.add_argument( "--top-n", type=int, default=TOP_N, help=( "Number of observations retained per probe per chronological bucket." ), )

    parser.add_argument( "--min-year", type=int, default=MIN_YEAR, )

    parser.add_argument( "--max-year", type=int, default=MAX_YEAR, )

    parser.add_argument( "--context", type=int, default=CONTEXT_TOKENS, help=( "Number of tokens of context either side of the retrieved token." ), )

    parser.add_argument( "--output", type=Path, default=DEFAULT_OUTPUT, help=( "Output HTML file." ), )

    args = parser.parse_args()

    if not 0 < args.distance < 2:
        parser.error( "--distance must be between 0 and 2" )

    if args.top_n < 1:
        parser.error( "--top-n must be at least 1" )

    if args.min_year > args.max_year:
        parser.error( "--min-year must not exceed --max-year" )

    logger.info( f"years: {args.min_year}–{args.max_year}" )

    logger.info( f"top per bucket/probe:  {args.top_n}" )

    logger.info( f"cluster distance:      {args.distance:.3f}" )

    logger.info( f"scale:                 {SCALE}" )

    logger.info( f"probes:                {sum(len(v) for v in PHRASE_PROBES.values())}" )

    # Initialise retrieval

    logger.info(" ")

    encoder = MacBertMeanPhraseEncoder()

    store = LanceObservationIndexStore( LANCE_INDEXES_DIR )

    search_space = SearchSpace(
        years=( args.min_year, args.max_year ),
        scale=(SCALE,),
    )

    # Search

    all_results = []

    for probe_group, phrases in PHRASE_PROBES.items():
        for phrase in phrases:
            logger.info( f"searching {probe_group}: {phrase!r}" )

            results = search_phrase(
                store,
                encoder,
                phrase,
                probe_group,
                search_space,
                args.top_n,
            )

            all_results.extend(results)

    # Merge duplicate observations

    observations = merge_results( all_results )

    logger.info(" ")
    logger.info( f"unique observations:  {len(observations)}" )

    # Metadata

    logger.info( "fetching PostgreSQL metadata..." )

    connection = get_connection()

    fetch_metadata( connection, observations )

    # Context

    logger.info( "fetching source context..." )

    for observation in observations:
        observation.context = fetch_context(
            connection,
            observation,
            args.context,
        )

        if (
            observation.pub_year is not None
            and observation.pub_year < 1200
        ):
            logger.info(
                "[pre-1200] %s (%s) token=%s event_id=%s\n%s",
                observation.title or observation.doc_id,
                observation.pub_year,
                observation.token,
                observation.event_id,
                observation.context,
            )


    # Vectors

    logger.info( "reconstructing Lance vectors..." )

    missing = fetch_vectors( store, observations, )

    if missing:
        logger.info( f"vectors unavailable:    {missing}" )

    observations = [
        observation
        for observation in observations
        if observation.vector is not None
    ]

    # Cluster

    bucket_clusters = assign_clusters(
        observations,
        args.distance,
    )

    print_summary( bucket_clusters )

    # PCA

    logger.info(" ")
    logger.info( "calculating PCA projections..." )

    assign_pca_coordinates( observations )

    # Figures

    figures = []

    for bucket in sorted(bucket_clusters):
        bucket_observations = [
            observation
            for observation in observations
            if observation.bucket == bucket
        ]

        figure = make_bucket_figure(
            bucket,
            bucket_observations,
            args.distance,
        )

        figures.append(
            (
                bucket,
                figure,
            )
        )

    # HTML

    logger.info(" ")
    logger.info( f"writing: {args.output}" )

    write_html(
        figures,
        args.output,
    )

    logger.info(" ")
    logger.info( "done" )


if __name__ == "__main__":
    main()
