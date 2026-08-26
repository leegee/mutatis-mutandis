from __future__ import annotations

import numpy as np
import umap
import igraph as ig
import leidenalg
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

from lib.corpus_logging import logger

MIN_IN_CLUSTER = 7

LOCAL_UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.05,
    "metric": "cosine",
}

GLOBAL_UMAP_PARAMS = {
    "n_neighbors": 50,
    "min_dist": 0.1,
    "metric": "cosine",
}

DEFAULT_GLOBAL_FIT_MAX = 50_000
DEFAULT_GLOBAL_TRANSFORM_BATCH = 10_000
DEFAULT_MIN_PER_STRATUM = 32

DEFAULT_COARSE_K = 16
DEFAULT_OUTLIER_FRACTION = 0.2

DEFAULT_EMBED_SUB_CHUNK = 20_000

DEFAULT_LOCAL_FIT_MAX = 50_000
DEFAULT_LOCAL_TRANSFORM_BATCH = 10_000


def load_event_rows(con, concept):
    """
    Load the empirical semantic field for a concept.

    The field is the union of:

      * seed events belonging directly to the concept;
      * distinct semantic neighbours retrieved from those seeds.

    Tier 2 stores neighbour metadata directly because a neighbour is an
    occurrence-level observation that may be reached from multiple seeds.

    Returns rows of:

        (event_id, vector_id, pub_year)
    """
    return con.execute(
        """
        SELECT
            event_id,
            vector_id,
            pub_year
        FROM events
        WHERE concept = ?

        UNION

        SELECT
            n.neighbour_event_id AS event_id,
            n.vector_id,
            n.pub_year
        FROM neighbours AS n
        JOIN events AS e
            ON e.event_id = n.event_id
        WHERE e.concept = ?

        ORDER BY event_id
        """,
        (
            concept,
            concept,
        ),
    ).fetchall()

def load_vectors(lookup, event_rows):
    event_ids = [
        int(row[0])
        for row in event_rows
    ]

    if not event_ids:
        return (
            [],
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
        )

    vectors = embeddings_year_major(
        lookup,
        event_ids,
    )

    return (
        event_ids,
        vectors,
    )


def project(vectors, params):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.zeros(
            (len(vectors), 2),
            dtype=np.float32,
        )

    reducer = umap.UMAP(
        random_state=42,
        **params,
    )

    return reducer.fit_transform(vectors)


def build_knn_graph(vectors, n_neighbors=15):
    if len(vectors) < 3:
        return []

    n_neighbors = min(
        n_neighbors,
        len(vectors) - 1,
    )

    if n_neighbors < 2:
        return []

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="cosine",
    )

    nn.fit(vectors)

    _, indices = nn.kneighbors(vectors)

    edges = []

    for idx, neighbours in enumerate(indices):
        for neighbour in neighbours[1:]:
            edges.append(
                (
                    idx,
                    int(neighbour),
                )
            )

    return edges


def leiden_cluster(
    vectors,
    resolution_parameter=0.8,
    n_neighbors=15,
):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.full(
            len(vectors),
            -1,
            dtype=np.int32,
        )

    edges = build_knn_graph(
        vectors,
        n_neighbors=n_neighbors,
    )

    graph = ig.Graph(
        n=len(vectors),
        edges=edges,
        directed=False,
    )

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        seed=42,
        resolution_parameter=resolution_parameter,
    )

    labels = np.full(
        len(vectors),
        -1,
        dtype=np.int32,
    )

    for cluster_id, members in enumerate(partition):
        for member in members:
            labels[member] = cluster_id

    return labels


def stratified_sample_ids(
    event_ids,
    strata,
    *,
    fit_max,
    min_per_stratum=DEFAULT_MIN_PER_STRATUM,
    seed=42,
):
    """
    Choose up to fit_max event ids with representative coverage.

    strata[i] is the stratum corresponding to event_ids[i].

    Each stratum receives a small floor when the budget permits, then
    remaining capacity is allocated approximately proportionally.
    """
    event_ids = [
        int(e)
        for e in event_ids
    ]

    n = len(event_ids)

    if n == 0 or fit_max <= 0:
        return []

    if n <= fit_max:
        return list(event_ids)

    rng = np.random.default_rng(seed)

    by_stratum = {}

    for eid, stratum in zip(
        event_ids,
        strata,
    ):
        by_stratum.setdefault(
            stratum,
            [],
        ).append(eid)

    labels = list(by_stratum.keys())

    sizes = np.array(
        [
            len(by_stratum[stratum])
            for stratum in labels
        ],
        dtype=np.int64,
    )

    n_strata = len(labels)

    floor = np.minimum(
        sizes,
        min_per_stratum,
    )

    if int(floor.sum()) > fit_max:
        floor = np.zeros(
            n_strata,
            dtype=np.int64,
        )

    remaining = int(
        fit_max - floor.sum()
    )

    if remaining > 0:
        weights = np.maximum(
            sizes - floor,
            0,
        )

        if weights.sum() > 0:
            extra = np.floor(
                remaining
                * (
                    weights
                    / weights.sum()
                )
            ).astype(np.int64)

            leftover = int(
                remaining
                - extra.sum()
            )

            order = np.argsort(
                -(
                    weights.astype(np.float64)
                    - extra
                )
            )

            for k in order[:leftover]:
                extra[k] += 1

            alloc = floor + extra

        else:
            alloc = floor

    else:
        alloc = floor

    alloc = np.minimum(
        alloc,
        sizes,
    )

    chosen = []

    for label, k in zip(
        labels,
        alloc,
    ):
        members = by_stratum[label]
        k = int(k)

        if k <= 0:
            continue

        if k >= len(members):
            chosen.extend(members)
        else:
            pick = rng.choice(
                len(members),
                size=k,
                replace=False,
            )

            chosen.extend(
                members[i]
                for i in pick
            )

    if len(chosen) < fit_max:
        chosen_set = set(chosen)

        rest = [
            event_id
            for event_id in event_ids
            if event_id not in chosen_set
        ]

        need = min(
            fit_max - len(chosen),
            len(rest),
        )

        if need > 0:
            pick = rng.choice(
                len(rest),
                size=need,
                replace=False,
            )

            chosen.extend(
                rest[i]
                for i in pick
            )

    rng.shuffle(chosen)

    return chosen


def iter_embeddings_year_major(
    lookup,
    event_ids,
    sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK,
):
    """
    Stream vector chunks grouped by publication year.

    The observation lookup owns physical vector storage. This layer does
    not know or care how vectors are indexed or persisted.

    Sub-chunking bounds the largest individual vector retrieval and
    temporary array, while year-major access preserves locality in the
    materialised Tier 1 store.
    """
    event_ids = [
        int(e)
        for e in event_ids
    ]

    if not event_ids:
        return

    by_year = {}

    for event_id in event_ids:
        pos = lookup.get_pos(event_id)
        year = int(
            lookup.pub_year[pos]
        )

        by_year.setdefault(
            year,
            [],
        ).append(event_id)

    for year in sorted(by_year):
        ids = by_year[year]

        logger.info(
            f"[tier3] embeddings year={year} "
            f"n={len(ids):,} "
            f"(of {len(event_ids):,} requested)"
        )

        for start in range(
            0,
            len(ids),
            sub_chunk_size,
        ):
            chunk_ids = ids[
                start:start + sub_chunk_size
            ]

            vectors = lookup.get_concatenated_embeddings(
                chunk_ids
            )

            vectors = np.asarray(
                vectors,
                dtype=np.float32,
            )

            yield (
                chunk_ids,
                vectors,
            )

            del vectors


def embeddings_year_major(
    lookup,
    event_ids,
    sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK,
):
    """
    Return vectors in the same order as event_ids.

    Retrieval is chunked, but this function intentionally materialises
    the final matrix because its callers have already bounded the number
    of events being fitted or transformed.
    """
    event_ids = [
        int(e)
        for e in event_ids
    ]

    if not event_ids:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    id_to_vec = {}

    for (
        chunk_ids,
        chunk_vectors,
    ) in iter_embeddings_year_major(
        lookup,
        event_ids,
        sub_chunk_size=sub_chunk_size,
    ):
        for i, event_id in enumerate(chunk_ids):
            id_to_vec[event_id] = chunk_vectors[i]

    first = next(
        iter(id_to_vec.values())
    )

    output = np.empty(
        (
            len(event_ids),
            first.shape[0],
        ),
        dtype=np.float32,
    )

    for i, event_id in enumerate(event_ids):
        output[i] = id_to_vec[event_id]

    return output


def fit_coarse_centroids(
    lookup,
    event_ids,
    *,
    k=DEFAULT_COARSE_K,
    sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK,
    seed=42,
):
    """
    Fit a small coarse clustering over the complete field.

    MiniBatchKMeans.partial_fit sees the complete field without requiring
    the complete field's embeddings to coexist in memory.
    """
    n = len(event_ids)
    k = max(
        1,
        min(k, n),
    )

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=3,
    )

    for (
        chunk_ids,
        chunk_vectors,
    ) in iter_embeddings_year_major(
        lookup,
        event_ids,
        sub_chunk_size=sub_chunk_size,
    ):
        kmeans.partial_fit(
            chunk_vectors
        )

    return kmeans.cluster_centers_.astype(
        np.float32
    )


def score_nearest_centroid_distance(
    lookup,
    event_ids,
    centroids,
    *,
    sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK,
):
    """
    Score every event by cosine distance to its nearest coarse centroid.

    Only O(n) scalar scores are retained; vectors are processed in chunks.
    """
    scores = {}

    for (
        chunk_ids,
        chunk_vectors,
    ) in iter_embeddings_year_major(
        lookup,
        event_ids,
        sub_chunk_size=sub_chunk_size,
    ):
        distances = cosine_distances(
            chunk_vectors,
            centroids,
        )

        nearest = distances.min(
            axis=1
        )

        for event_id, distance in zip(
            chunk_ids,
            nearest,
        ):
            scores[event_id] = float(
                distance
            )

    return scores


def select_local_fit_sample(
    event_ids,
    distances,
    strata,
    *,
    fit_max,
    outlier_fraction=DEFAULT_OUTLIER_FRACTION,
    min_per_stratum=DEFAULT_MIN_PER_STRATUM,
    seed=42,
):
    """
    Select a bounded local fit sample while guaranteeing inclusion of
    the most anomalous events.

    The remaining capacity is filled by representative stratified
    sampling.
    """
    event_ids = [
        int(e)
        for e in event_ids
    ]

    n = len(event_ids)

    if n <= fit_max:
        return (
            list(event_ids),
            [],
        )

    outlier_budget = max(
        0,
        min(
            int(
                fit_max
                * outlier_fraction
            ),
            fit_max,
        ),
    )

    ranked = sorted(
        event_ids,
        key=lambda eid: distances.get(
            eid,
            0.0,
        ),
        reverse=True,
    )

    outlier_ids = ranked[
        :outlier_budget
    ]

    outlier_set = set(
        outlier_ids
    )

    remaining_budget = (
        fit_max
        - len(outlier_ids)
    )

    rest_ids = []
    rest_strata = []

    for event_id, stratum in zip(
        event_ids,
        strata,
    ):
        if event_id not in outlier_set:
            rest_ids.append(event_id)
            rest_strata.append(stratum)

    core_ids = stratified_sample_ids(
        rest_ids,
        rest_strata,
        fit_max=remaining_budget,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    fit_ids = (
        outlier_ids
        + core_ids
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(fit_ids)

    return (
        fit_ids,
        outlier_ids,
    )


def build_global_projection(
    lookup,
    all_field_event_ids,
    *,
    strata=None,
    fit_max=DEFAULT_GLOBAL_FIT_MAX,
    transform_batch=DEFAULT_GLOBAL_TRANSFORM_BATCH,
    min_per_stratum=DEFAULT_MIN_PER_STRATUM,
    seed=42,
):
    """
    Map every field event_id to 2D global coordinates.

    Large fields are represented by a bounded stratified UMAP fit sample;
    the remainder is transformed in batches.
    """
    event_ids = [
        int(e)
        for e in all_field_event_ids
    ]

    n = len(event_ids)

    logger.info(
        f"[tier3] global projection "
        f"events={n:,} "
        f"fit_max={fit_max:,}"
    )

    if n == 0:
        return {}

    if n < MIN_IN_CLUSTER:
        return {
            event_id: np.zeros(
                2,
                dtype=np.float32,
            )
            for event_id in event_ids
        }

    if n <= fit_max:
        vectors = embeddings_year_major(
            lookup,
            event_ids,
        )

        coords = project(
            vectors,
            GLOBAL_UMAP_PARAMS,
        )

        return {
            event_id: coords[idx].astype(
                np.float32
            )
            for idx, event_id
            in enumerate(event_ids)
        }

    if strata is None:
        logger.warning(
            "[tier3] global projection: "
            "no strata provided; using uniform sample"
        )

        strata = [
            "_all"
        ] * n

    if len(strata) != n:
        raise ValueError(
            "strata length must match "
            "all_field_event_ids"
        )

    fit_ids = stratified_sample_ids(
        event_ids,
        strata,
        fit_max=fit_max,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    n_strata = len(
        set(strata)
    )

    logger.info(
        f"[tier3] global UMAP fit on "
        f"{len(fit_ids):,} / {n:,} events "
        f"across ~{n_strata} strata; "
        f"transform remainder in batches of "
        f"{transform_batch:,}"
    )

    fit_vectors = embeddings_year_major(
        lookup,
        fit_ids,
    )

    reducer = umap.UMAP(
        random_state=seed,
        **GLOBAL_UMAP_PARAMS,
    )

    fit_coords = reducer.fit_transform(
        fit_vectors
    )

    out = {
        event_id: fit_coords[j].astype(
            np.float32
        )
        for j, event_id
        in enumerate(fit_ids)
    }

    del fit_vectors

    fit_set = set(fit_ids)

    rest_ids = [
        event_id
        for event_id in event_ids
        if event_id not in fit_set
    ]

    for start in range(
        0,
        len(rest_ids),
        transform_batch,
    ):
        batch_ids = rest_ids[
            start:start + transform_batch
        ]

        batch_vectors = embeddings_year_major(
            lookup,
            batch_ids,
        )

        batch_coords = reducer.transform(
            batch_vectors
        )

        for j, event_id in enumerate(batch_ids):
            out[event_id] = (
                batch_coords[j]
                .astype(np.float32)
            )

        del batch_vectors

        if (
            start // transform_batch
        ) % 5 == 0:
            logger.info(
                f"[tier3] global transform "
                f"{min(start + transform_batch, len(rest_ids)):,}"
                f"/{len(rest_ids):,}"
            )

    return out


def compute_cluster_centroids(
    vectors,
    local_coords,
    global_coords,
    clusters,
):
    records = []

    for cluster_id in sorted(
        set(clusters)
    ):
        mask = (
            clusters == cluster_id
        )

        if not np.any(mask):
            continue

        records.append(
            {
                "cluster_id": int(
                    cluster_id
                ),
                "centroid_vector": (
                    vectors[mask]
                    .mean(axis=0)
                    .astype(np.float32)
                ),
                "centroid_nx": float(
                    local_coords[
                        mask,
                        0,
                    ].mean()
                ),
                "centroid_ny": float(
                    local_coords[
                        mask,
                        1,
                    ].mean()
                ),
                "centroid_gnx": float(
                    global_coords[
                        mask,
                        0,
                    ].mean()
                ),
                "centroid_gny": float(
                    global_coords[
                        mask,
                        1,
                    ].mean()
                ),
                "point_count": int(
                    mask.sum()
                ),
            }
        )

    return records


def compute_cluster_centroid_vectors_streaming(
    lookup,
    event_ids,
    clusters,
    *,
    sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK,
):
    """
    Compute per-cluster mean vectors without materialising the complete
    field's embedding matrix.
    """
    event_ids = [
        int(e)
        for e in event_ids
    ]

    cluster_by_id = {
        event_id: int(cluster)
        for event_id, cluster
        in zip(
            event_ids,
            clusters,
        )
    }

    sums = {}
    counts = {}

    for (
        chunk_ids,
        chunk_vectors,
    ) in iter_embeddings_year_major(
        lookup,
        event_ids,
        sub_chunk_size=sub_chunk_size,
    ):
        for event_id, vector in zip(
            chunk_ids,
            chunk_vectors,
        ):
            cluster_id = cluster_by_id[
                event_id
            ]

            if cluster_id not in sums:
                sums[cluster_id] = np.zeros_like(
                    vector,
                    dtype=np.float64,
                )
                counts[cluster_id] = 0

            sums[cluster_id] += vector
            counts[cluster_id] += 1

    return {
        cluster_id: (
            sums[cluster_id]
            / counts[cluster_id]
        ).astype(np.float32)
        for cluster_id in sums
    }


def assign_labels_by_nearest_fit(
    fit_vectors,
    fit_labels,
    other_vectors,
):
    """
    Propagate fitted Leiden labels to points outside the fit sample by
    nearest neighbour in the original embedding space.
    """
    if len(other_vectors) == 0:
        return np.empty(
            (0,),
            dtype=np.int32,
        )

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
    )

    nn.fit(fit_vectors)

    _, indices = nn.kneighbors(
        other_vectors
    )

    return np.asarray(
        [
            fit_labels[i[0]]
            for i in indices
        ],
        dtype=np.int32,
    )


def local_project_and_cluster(
    lookup,
    event_ids,
    *,
    strata=None,
    umap_params=None,
    resolution_parameter=0.8,
    n_neighbors=15,
    fit_max=DEFAULT_LOCAL_FIT_MAX,
    transform_batch=DEFAULT_LOCAL_TRANSFORM_BATCH,
    outlier_fraction=DEFAULT_OUTLIER_FRACTION,
    coarse_k=DEFAULT_COARSE_K,
    min_per_stratum=DEFAULT_MIN_PER_STRATUM,
    seed=42,
):
    """
    Project and Leiden-cluster one concept's semantic field.

    Small fields are fitted directly.

    Large fields are processed using:
      1. streaming coarse clustering;
      2. streaming anomaly scoring;
      3. bounded outlier-guaranteed sampling;
      4. UMAP + Leiden on the bounded sample;
      5. batched UMAP transformation and nearest-fit label propagation;
      6. streaming centroid calculation over the complete field.

    No complete large-field embedding matrix is required.
    """
    if umap_params is None:
        umap_params = LOCAL_UMAP_PARAMS

    event_ids = [
        int(e)
        for e in event_ids
    ]

    n = len(event_ids)

    if n < MIN_IN_CLUSTER:
        clusters = np.full(
            n,
            -1,
            dtype=np.int32,
        )

        return {
            "event_ids": event_ids,
            "local_coords": np.zeros(
                (n, 2),
                dtype=np.float32,
            ),
            "clusters": clusters,
            "cluster_centroid_vectors": {},
            "fit_info": {
                "n": n,
                "fit_n": n,
                "outlier_n": 0,
                "sampled": False,
            },
        }

    if n <= fit_max:
        vectors = embeddings_year_major(
            lookup,
            event_ids,
        )

        local_coords = project(
            vectors,
            umap_params,
        )

        clusters = leiden_cluster(
            vectors,
            resolution_parameter=resolution_parameter,
            n_neighbors=n_neighbors,
        )

        cluster_centroid_vectors = {
            int(cluster_id):
                vectors[
                    clusters == cluster_id
                ]
                .mean(axis=0)
                .astype(np.float32)
            for cluster_id in sorted(
                set(
                    int(c)
                    for c in clusters
                )
            )
        }

        return {
            "event_ids": event_ids,
            "local_coords": local_coords,
            "clusters": clusters,
            "cluster_centroid_vectors": (
                cluster_centroid_vectors
            ),
            "fit_info": {
                "n": n,
                "fit_n": n,
                "outlier_n": 0,
                "sampled": False,
            },
        }

    if strata is None:
        logger.warning(
            "[tier3] local projection: "
            "no strata provided; using uniform "
            "representative sampling"
        )

        strata = [
            "_all"
        ] * n

    logger.info(
        f"[tier3] local field n={n:,} "
        f"exceeds fit_max={fit_max:,}; "
        f"fitting coarse centroids "
        f"(k={coarse_k}) to score outliers"
    )

    centroids = fit_coarse_centroids(
        lookup,
        event_ids,
        k=coarse_k,
        seed=seed,
    )

    distances = score_nearest_centroid_distance(
        lookup,
        event_ids,
        centroids,
    )

    fit_ids, outlier_ids = (
        select_local_fit_sample(
            event_ids,
            distances,
            strata,
            fit_max=fit_max,
            outlier_fraction=outlier_fraction,
            min_per_stratum=min_per_stratum,
            seed=seed,
        )
    )

    logger.info(
        f"[tier3] local UMAP fit on "
        f"{len(fit_ids):,} / {n:,} events "
        f"({len(outlier_ids):,} guaranteed as outliers); "
        f"transform remainder in batches of "
        f"{transform_batch:,}"
    )

    fit_vectors = embeddings_year_major(
        lookup,
        fit_ids,
    )

    reducer = umap.UMAP(
        random_state=seed,
        **umap_params,
    )

    fit_coords = reducer.fit_transform(
        fit_vectors
    )

    fit_clusters = leiden_cluster(
        fit_vectors,
        resolution_parameter=resolution_parameter,
        n_neighbors=n_neighbors,
    )

    coords_by_id = {
        event_id: fit_coords[j].astype(
            np.float32
        )
        for j, event_id
        in enumerate(fit_ids)
    }

    clusters_by_id = {
        event_id: int(
            fit_clusters[j]
        )
        for j, event_id
        in enumerate(fit_ids)
    }

    fit_set = set(fit_ids)

    rest_ids = [
        event_id
        for event_id in event_ids
        if event_id not in fit_set
    ]

    for start in range(
        0,
        len(rest_ids),
        transform_batch,
    ):
        batch_ids = rest_ids[
            start:start + transform_batch
        ]

        batch_vectors = embeddings_year_major(
            lookup,
            batch_ids,
        )

        batch_coords = reducer.transform(
            batch_vectors
        )

        batch_clusters = (
            assign_labels_by_nearest_fit(
                fit_vectors,
                fit_clusters,
                batch_vectors,
            )
        )

        for j, event_id in enumerate(batch_ids):
            coords_by_id[event_id] = (
                batch_coords[j]
                .astype(np.float32)
            )

            clusters_by_id[event_id] = int(
                batch_clusters[j]
            )

        del batch_vectors

    del fit_vectors

    local_coords = np.asarray(
        [
            coords_by_id[event_id]
            for event_id in event_ids
        ],
        dtype=np.float32,
    )

    clusters = np.asarray(
        [
            clusters_by_id[event_id]
            for event_id in event_ids
        ],
        dtype=np.int32,
    )

    cluster_centroid_vectors = (
        compute_cluster_centroid_vectors_streaming(
            lookup,
            event_ids,
            clusters,
        )
    )

    return {
        "event_ids": event_ids,
        "local_coords": local_coords,
        "clusters": clusters,
        "cluster_centroid_vectors": (
            cluster_centroid_vectors
        ),
        "fit_info": {
            "n": n,
            "fit_n": len(fit_ids),
            "outlier_n": len(outlier_ids),
            "sampled": True,
        },
    }
