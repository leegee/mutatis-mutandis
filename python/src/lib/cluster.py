# lib/clustering.py

from __future__ import annotations

import numpy as np
import umap
import igraph as ig
import leidenalg
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances

from lib.corpus_logging import logger
from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    ZARR_PATH,
    faiss_index_paths,
)


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



def load_event_rows(con, concept):
    """
    Load the empirical semantic field for a concept.

    concept_field_events is the authoritative relation:
        concept -> observed corpus events

    Seed events anchor the field.
    Neighbour events are retrieved semantic context.

    An event may belong to multiple fields.

    Returns rows of (event_id, vector_id, pub_year).
    """
    return con.execute(
        """
        SELECT e.event_id, e.vector_id, e.pub_year
        FROM concept_field_events f
        JOIN events e ON e.event_id = f.event_id
        WHERE f.concept = ?
        ORDER BY e.event_id
        """,
        (concept,),
    ).fetchall()


def load_vectors( lookup, event_rows, ):
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

    # Year-major + evict so a large concept field cannot pin every
    # year's FAISS indices at once.
    vectors = embeddings_year_major(lookup, event_ids)

    return ( event_ids, vectors, )


def project( vectors, params, ):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.zeros( ( len(vectors), 2, ), dtype=np.float32, )
    reducer = umap.UMAP( random_state=42, **params, )
    return reducer.fit_transform( vectors )


def build_knn_graph( vectors, n_neighbors=15, ):
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

    _, indices = nn.kneighbors(
        vectors
    )

    edges = []

    for idx, neighbours in enumerate(indices):
        for neighbour in neighbours[1:]:
            edges.append( (
                idx,
                int(neighbour),
            ) )

    return edges


def leiden_cluster( vectors, resolution_parameter=0.8, n_neighbors=15 ):
    if len(vectors) < MIN_IN_CLUSTER:
        return np.full( len(vectors), -1, dtype=np.int32, )

    edges = build_knn_graph( vectors, n_neighbors=n_neighbors )

    graph = ig.Graph(
        n=len(vectors),      # explicit vertex count, no longer inferred from edges
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


# Cap for fitting the global UMAP. Above this, fit on a representative
# sample and UMAP.transform the remainder in batches so peak RAM stays
# bounded.
DEFAULT_GLOBAL_FIT_MAX = 50_000
DEFAULT_GLOBAL_TRANSFORM_BATCH = 10_000
# Floor per stratum so rare concepts/periods are not zeroed out when
# proportional allocation would round to 0.
DEFAULT_MIN_PER_STRATUM = 32


def stratified_sample_ids(
    event_ids,
    strata,
    *,
    fit_max: int,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
    seed: int = 42,
):
    """
    Choose up to `fit_max` event ids with representative coverage.

    `strata[i]` is the stratum label for `event_ids[i]` (e.g. concept,
    or (concept, century)). Each stratum gets at least
    min(min_per_stratum, size) slots when budget allows, then remaining
    slots are allocated proportional to stratum size.
    """
    event_ids = [int(e) for e in event_ids]
    n = len(event_ids)
    if n == 0 or fit_max <= 0:
        return []
    if n <= fit_max:
        return list(event_ids)

    rng = np.random.default_rng(seed)

    by_stratum = {}
    for eid, s in zip(event_ids, strata):
        by_stratum.setdefault(s, []).append(eid)

    labels = list(by_stratum.keys())
    sizes = np.array([len(by_stratum[s]) for s in labels], dtype=np.int64)
    n_strata = len(labels)

    floor = np.minimum(sizes, min_per_stratum)
    if int(floor.sum()) > fit_max:
        floor = np.zeros(n_strata, dtype=np.int64)

    remaining = int(fit_max - floor.sum())
    if remaining > 0:
        weights = np.maximum(sizes - floor, 0)
        if weights.sum() > 0:
            extra = np.floor(remaining * (weights / weights.sum())).astype(np.int64)
            leftover = int(remaining - extra.sum())
            order = np.argsort(-(weights.astype(np.float64) - extra))
            for k in order[:leftover]:
                extra[k] += 1
            alloc = floor + extra
        else:
            alloc = floor
    else:
        alloc = floor

    alloc = np.minimum(alloc, sizes)

    chosen = []
    for label, k in zip(labels, alloc):
        members = by_stratum[label]
        k = int(k)
        if k <= 0:
            continue
        if k >= len(members):
            chosen.extend(members)
        else:
            pick = rng.choice(len(members), size=k, replace=False)
            chosen.extend(members[i] for i in pick)

    if len(chosen) < fit_max:
        chosen_set = set(chosen)
        rest = [e for e in event_ids if e not in chosen_set]
        need = min(fit_max - len(chosen), len(rest))
        if need > 0:
            pick = rng.choice(len(rest), size=need, replace=False)
            chosen.extend(rest[i] for i in pick)

    rng.shuffle(chosen)
    return chosen



def _find_lazy_indexes(lookup):
    for attr in ("_index", "index", "_indexes", "indexes"):
        cand = getattr(lookup, attr, None)
        if cand is not None and hasattr(cand, "evict"):
            return cand
    return None


# Sub-chunk size within a single publication year when streaming
# embeddings for a concept. Keeps any single reconstruction call (and
# any array built from it) bounded even when one year dominates a
# concept's field — e.g. a pamphlet boom year. Deliberately generous
# relative to typical per-chunk costs elsewhere in tier2/tier3, since
# it only needs to bound *this* call's memory, not drive algorithm
# behaviour the way tier2's BATCH_SIZE does.
DEFAULT_EMBED_SUB_CHUNK = 20_000


def iter_embeddings_year_major(lookup, event_ids, sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK, indexes=None):
    """
    Stream (ids_chunk, vectors_chunk) pairs for event_ids, year-major,
    sub-chunked within a year, without ever materialising a full
    n_events x dim matrix.

    Groups ids by publication year (as embeddings_year_major does), but
    additionally sub-chunks within a year so a single dominant year
    (e.g. a pamphlet boom) can't itself force a huge reconstruction or
    a huge chunk array. A year's FAISS index is evicted (if `indexes`
    supports it) only once, after every sub-chunk for that year has
    been yielded — not per sub-chunk, since that would just force
    repeated reloads of the same year's index.

    This is the shared primitive behind embeddings_year_major (which
    still returns one full matrix, for callers that have already
    bounded n) and the streaming outlier-scoring functions below (which
    never need — and deliberately never build — a full matrix at all).
    """
    event_ids = [int(e) for e in event_ids]
    if not event_ids:
        return

    by_year = {}
    for eid in event_ids:
        pos = lookup.get_pos(eid)
        year = int(lookup.pub_year[pos])
        by_year.setdefault(year, []).append(eid)

    if indexes is None:
        indexes = _find_lazy_indexes(lookup)

    for year in sorted(by_year):
        ids = by_year[year]
        logger.info(
            f"[tier3] embeddings year={year} n={len(ids):,} "
            f"(of {len(event_ids):,} requested)"
        )

        for start in range(0, len(ids), sub_chunk_size):
            chunk_ids = ids[start:start + sub_chunk_size]
            vecs = lookup.get_concatenated_embeddings(chunk_ids)
            vecs = np.asarray(vecs, dtype=np.float32)
            yield chunk_ids, vecs
            del vecs

        if indexes is not None and hasattr(indexes, "evict"):
            indexes.evict(year)


def embeddings_year_major(lookup, event_ids, sub_chunk_size=DEFAULT_EMBED_SUB_CHUNK):
    """
    Load concatenated embeddings for event_ids without holding every
    year's FAISS index at once.

    Returns one full (n_events, dim) matrix, in the same order as
    `event_ids` — for callers that have already bounded n (e.g. a
    UMAP fit sample). For unbounded concept fields, prefer streaming
    via iter_embeddings_year_major directly (see fit_coarse_centroids /
    score_nearest_centroid_distance below), which never materialises
    the full matrix at all.
    """
    event_ids = [int(e) for e in event_ids]
    if not event_ids:
        return np.empty((0, 0), dtype=np.float32)

    id_to_vec = {}
    for chunk_ids, chunk_vecs in iter_embeddings_year_major(lookup, event_ids, sub_chunk_size=sub_chunk_size):
        for i, eid in enumerate(chunk_ids):
            id_to_vec[eid] = chunk_vecs[i]

    dim = next(iter(id_to_vec.values())).shape[0]
    out = np.empty((len(event_ids), dim), dtype=np.float32)
    for i, eid in enumerate(event_ids):
        out[i] = id_to_vec[eid]
    return out


# ----------------------------------------------------------------------
# Outlier-aware fit sampling for the LOCAL (per-concept) projection.
#
# stratified_sample_ids above optimises for *representative coverage*
# — it explicitly avoids under-representing rare strata, but has no
# notion of "this individual point is unusual." For polysemy detection,
# where the whole point is to find anomalous individual events, that's
# the wrong sampling goal on its own: a genuine outlier could easily
# lose a coin-flip against a well-populated stratum and never make the
# UMAP fit sample, after which .transform() would (by construction)
# pull it toward whatever's nearest in the *fitted* manifold rather
# than showing it as the outlier it is.
#
# The fix: score every point's "how well is this explained by the
# concept's common usage" via distance to the nearest of a small number
# of coarse sense-clusters, computed streaming (no full matrix needed),
# then GUARANTEE the highest-scoring points are in the fit sample and
# only fill the remaining budget with the existing representative
# sample. Only genuinely unremarkable points get .transform()-ed.
# ----------------------------------------------------------------------

DEFAULT_COARSE_K = 16
DEFAULT_OUTLIER_FRACTION = 0.2


def fit_coarse_centroids(
    lookup,
    event_ids,
    *,
    k: int = DEFAULT_COARSE_K,
    sub_chunk_size: int = DEFAULT_EMBED_SUB_CHUNK,
    seed: int = 42,
    indexes=None,
):
    """
    Fit a small, cheap coarse clustering over a concept's full field,
    streaming — used only as an anomaly yardstick ("how far is this
    point from the nearest common usage pattern"), not as the actual
    semantic clustering (that's still Leiden, on the bounded fit
    sample). MiniBatchKMeans.partial_fit lets this see every point
    without ever holding more than one chunk's vectors at a time.

    Returns the fitted centroids as a (k, dim) float32 array.
    """
    n = len(event_ids)
    k = max(1, min(k, n))

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3)

    for chunk_ids, chunk_vecs in iter_embeddings_year_major(
        lookup, event_ids, sub_chunk_size=sub_chunk_size, indexes=indexes
    ):
        kmeans.partial_fit(chunk_vecs)

    return kmeans.cluster_centers_.astype(np.float32)


def score_nearest_centroid_distance(
    lookup,
    event_ids,
    centroids,
    *,
    sub_chunk_size: int = DEFAULT_EMBED_SUB_CHUNK,
    indexes=None,
):
    """
    Streaming anomaly score for every event: cosine distance to the
    NEAREST coarse centroid (not the single overall mean — a point on
    the far side of a legitimate second common sense is not what we're
    trying to flag; a point poorly explained by *any* common sense is).

    Returns {event_id: distance}. Memory cost is O(n) floats (plus one
    chunk of vectors at a time), never O(n * dim).
    """
    scores = {}
    for chunk_ids, chunk_vecs in iter_embeddings_year_major(
        lookup, event_ids, sub_chunk_size=sub_chunk_size, indexes=indexes
    ):
        dists = cosine_distances(chunk_vecs, centroids)
        nearest = dists.min(axis=1)
        for eid, d in zip(chunk_ids, nearest):
            scores[eid] = float(d)
    return scores


def select_local_fit_sample(
    event_ids,
    distances,
    strata,
    *,
    fit_max: int,
    outlier_fraction: float = DEFAULT_OUTLIER_FRACTION,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
    seed: int = 42,
):
    """
    Choose up to `fit_max` event ids for the local UMAP+Leiden fit,
    guaranteeing the most anomalous points (by `distances`, e.g. from
    score_nearest_centroid_distance) are included, and filling the
    remaining budget with a representative stratified sample of what's
    left — so common usage still gets fair coverage, but no outlier is
    ever *only* eligible for the sampling lottery.

    Returns (fit_ids, outlier_ids) — outlier_ids is the subset of
    fit_ids that were included specifically for being anomalous, useful
    for logging / for flagging them in downstream output.
    """
    event_ids = [int(e) for e in event_ids]
    n = len(event_ids)
    if n <= fit_max:
        return list(event_ids), []

    outlier_budget = max(0, min(int(fit_max * outlier_fraction), fit_max))

    ranked = sorted(event_ids, key=lambda eid: distances.get(eid, 0.0), reverse=True)
    outlier_ids = ranked[:outlier_budget]
    outlier_set = set(outlier_ids)

    remaining_budget = fit_max - len(outlier_ids)
    rest_ids = []
    rest_strata = []
    for eid, s in zip(event_ids, strata):
        if eid not in outlier_set:
            rest_ids.append(eid)
            rest_strata.append(s)

    core_ids = stratified_sample_ids(
        rest_ids,
        rest_strata,
        fit_max=remaining_budget,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    fit_ids = outlier_ids + core_ids
    rng = np.random.default_rng(seed)
    rng.shuffle(fit_ids)
    return fit_ids, outlier_ids


def build_global_projection(
    lookup,
    all_field_event_ids,
    *,
    strata=None,
    fit_max: int = DEFAULT_GLOBAL_FIT_MAX,
    transform_batch: int = DEFAULT_GLOBAL_TRANSFORM_BATCH,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
    seed: int = 42,
):
    """
    Map every field event_id → 2D global coordinates.

    For large fields, fit UMAP on up to `fit_max` *representative*
    points (stratified when `strata` is provided), then transform the
    rest in batches.

    `strata`: optional sequence aligned with `all_field_event_ids`
    (e.g. concept name, or (concept, century)). Without it, sampling
    is uniform — avoid that when concept sizes are highly skewed.
    """
    event_ids = [int(e) for e in all_field_event_ids]
    n = len(event_ids)
    logger.info(f"[tier3] global projection events={n:,} fit_max={fit_max:,}")

    if n == 0:
        return {}

    if n < MIN_IN_CLUSTER:
        return {eid: np.zeros(2, dtype=np.float32) for eid in event_ids}

    if n <= fit_max:
        vectors = embeddings_year_major(lookup, event_ids)
        coords = project(vectors, GLOBAL_UMAP_PARAMS)
        return {
            eid: coords[idx].astype(np.float32)
            for idx, eid in enumerate(event_ids)
        }

    if strata is None:
        logger.warning(
            "[tier3] global projection: no strata provided; "
            "using uniform sample (may under-represent rare concepts)"
        )
        strata = ["_all"] * n

    if len(strata) != n:
        raise ValueError("strata length must match all_field_event_ids")

    fit_ids = stratified_sample_ids(
        event_ids,
        strata,
        fit_max=fit_max,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    n_strata = len(set(strata))
    logger.info(
        f"[tier3] global UMAP fit on {len(fit_ids):,} / {n:,} events "
        f"across ~{n_strata} strata; "
        f"transform remainder in batches of {transform_batch:,}"
    )

    fit_vectors = embeddings_year_major(lookup, fit_ids)
    reducer = umap.UMAP(random_state=seed, **GLOBAL_UMAP_PARAMS)
    fit_coords = reducer.fit_transform(fit_vectors)

    out = {
        eid: fit_coords[j].astype(np.float32)
        for j, eid in enumerate(fit_ids)
    }
    del fit_vectors

    fit_set = set(fit_ids)
    rest_ids = [eid for eid in event_ids if eid not in fit_set]

    for start in range(0, len(rest_ids), transform_batch):
        batch_ids = rest_ids[start : start + transform_batch]
        batch_vectors = embeddings_year_major(lookup, batch_ids)
        batch_coords = reducer.transform(batch_vectors)
        for j, eid in enumerate(batch_ids):
            out[eid] = batch_coords[j].astype(np.float32)
        del batch_vectors

        if (start // transform_batch) % 5 == 0:
            logger.info(
                f"[tier3] global transform "
                f"{min(start + transform_batch, len(rest_ids)):,}/{len(rest_ids):,}"
            )

    return out


def compute_cluster_centroids(
    vectors,
    local_coords,
    global_coords,
    clusters,
):
    records = []

    for cluster_id in sorted(set(clusters)):
        mask = clusters == cluster_id

        if not np.any(mask):
            continue

        records.append( {
            "cluster_id": int(cluster_id),
            "centroid_vector": vectors[mask] .mean(axis=0) .astype(np.float32),
            "centroid_nx": float(local_coords[mask,0].mean()),
            "centroid_ny": float(local_coords[mask,1].mean()),
            "centroid_gnx": float(global_coords[mask,0].mean()),
            "centroid_gny": float(global_coords[mask,1].mean()),
            "point_count": int(mask.sum()),
        } )

    return records


# ----------------------------------------------------------------------
# Bounded LOCAL (per-concept) projection + clustering.
#
# The counterpart to build_global_projection above, for the per-concept
# UMAP+Leiden step in tier3_project_cluster.cluster_concept. Small
# concepts take an unchanged fast path (fit on everything, exactly as
# before). Large concepts (concept_field_events can run into the
# hundreds of thousands — seeds plus up to K neighbours each) fit on a
# bounded, outlier-guaranteed sample and transform/label-propagate the
# rest, instead of ever materialising the full field's embedding matrix
# or a full-field k-NN graph.
# ----------------------------------------------------------------------

DEFAULT_LOCAL_FIT_MAX = 50_000
DEFAULT_LOCAL_TRANSFORM_BATCH = 10_000


def compute_cluster_centroid_vectors_streaming(
    lookup,
    event_ids,
    clusters,
    *,
    sub_chunk_size: int = DEFAULT_EMBED_SUB_CHUNK,
    indexes=None,
):
    """
    Per-cluster mean embedding, computed streaming (running sum/count
    per cluster_id) instead of requiring the full (n, dim) matrix in
    memory. Returns {cluster_id: centroid_vector (float32)}.
    """
    event_ids = [int(e) for e in event_ids]
    cluster_by_id = {eid: int(c) for eid, c in zip(event_ids, clusters)}

    sums = {}
    counts = {}

    for chunk_ids, chunk_vecs in iter_embeddings_year_major(
        lookup, event_ids, sub_chunk_size=sub_chunk_size, indexes=indexes
    ):
        for eid, vec in zip(chunk_ids, chunk_vecs):
            cid = cluster_by_id[eid]
            if cid not in sums:
                sums[cid] = np.zeros_like(vec, dtype=np.float64)
                counts[cid] = 0
            sums[cid] += vec
            counts[cid] += 1

    return {
        cid: (sums[cid] / counts[cid]).astype(np.float32)
        for cid in sums
    }


def assign_labels_by_nearest_fit(fit_vectors, fit_labels, other_vectors):
    """
    Propagate integer labels (e.g. Leiden cluster ids) from a fitted
    sample to additional points, via 1-nearest-neighbour in embedding
    space (cosine). Used for the "transformed, not fitted" remainder of
    a large concept's field — cheap relative to re-running Leiden on
    everything, and reasonable since these are, by construction, the
    points score_nearest_centroid_distance found unremarkable (i.e.
    well explained by existing structure), not held-out outliers.
    """
    if len(other_vectors) == 0:
        return np.empty((0,), dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(fit_vectors)
    _, indices = nn.kneighbors(other_vectors)
    return np.asarray(
        [fit_labels[i[0]] for i in indices],
        dtype=np.int32,
    )


def local_project_and_cluster(
    lookup,
    event_ids,
    *,
    strata=None,
    umap_params=None,
    resolution_parameter: float = 0.8,
    n_neighbors: int = 15,
    fit_max: int = DEFAULT_LOCAL_FIT_MAX,
    transform_batch: int = DEFAULT_LOCAL_TRANSFORM_BATCH,
    outlier_fraction: float = DEFAULT_OUTLIER_FRACTION,
    coarse_k: int = DEFAULT_COARSE_K,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
    seed: int = 42,
):
    """
    Bounded replacement for calling project(vectors, LOCAL_UMAP_PARAMS)
    + leiden_cluster(vectors, ...) directly on a concept's full field.

    n <= fit_max: unchanged behaviour — load everything, fit UMAP and
    Leiden on the whole field, exactly as before.

    n > fit_max:
      1. Fit a small coarse clustering streaming (fit_coarse_centroids)
         — never materialises the full field's embedding matrix.
      2. Score every event's distance to its nearest coarse centroid
         (score_nearest_centroid_distance) — streaming, O(n) floats.
      3. Build a bounded fit sample that GUARANTEES the top
         `outlier_fraction` most-anomalous points are included
         (select_local_fit_sample), filling the rest with a
         representative stratified sample.
      4. Load only the fit sample's vectors (bounded to `fit_max`),
         run the real UMAP + Leiden clustering on them.
      5. For the remainder: transform() into the fitted UMAP space in
         batches, and propagate cluster labels via nearest-fit-point
         (assign_labels_by_nearest_fit). These are exactly the points
         already found to be well-explained by common structure, so
         transform's "pulls toward what's already there" behaviour is
         appropriate here — the actual outliers were never left to it.
      6. Cluster centroid vectors are computed streaming over the WHOLE
         field (compute_cluster_centroid_vectors_streaming), not just
         the fit sample, so downstream centroid/description data
         reflects every event, not only the ones UMAP was fit on.

    Returns a dict:
        event_ids: list[int]              (same order as input)
        local_coords: (n, 2) float32
        clusters: (n,) int32
        cluster_centroid_vectors: {cluster_id: (dim,) float32}
        fit_info: {"n": n, "fit_n": ..., "outlier_n": ..., "sampled": bool}
    """
    if umap_params is None:
        umap_params = LOCAL_UMAP_PARAMS

    event_ids = [int(e) for e in event_ids]
    n = len(event_ids)

    if n < MIN_IN_CLUSTER:
        clusters = np.full(n, -1, dtype=np.int32)
        return {
            "event_ids": event_ids,
            "local_coords": np.zeros((n, 2), dtype=np.float32),
            "clusters": clusters,
            "cluster_centroid_vectors": {},
            "fit_info": {"n": n, "fit_n": n, "outlier_n": 0, "sampled": False},
        }

    if n <= fit_max:
        vectors = embeddings_year_major(lookup, event_ids)
        local_coords = project(vectors, umap_params)
        clusters = leiden_cluster(
            vectors, resolution_parameter=resolution_parameter, n_neighbors=n_neighbors
        )
        cluster_centroid_vectors = {
            int(cid): vectors[clusters == cid].mean(axis=0).astype(np.float32)
            for cid in sorted(set(int(c) for c in clusters))
        }
        return {
            "event_ids": event_ids,
            "local_coords": local_coords,
            "clusters": clusters,
            "cluster_centroid_vectors": cluster_centroid_vectors,
            "fit_info": {"n": n, "fit_n": n, "outlier_n": 0, "sampled": False},
        }

    # --- Large field: bounded, outlier-guaranteed sampling path ---
    if strata is None:
        logger.warning(
            "[tier3] local projection: no strata provided; "
            "using uniform sample for the representative portion"
        )
        strata = ["_all"] * n

    logger.info(
        f"[tier3] local field n={n:,} exceeds fit_max={fit_max:,}; "
        f"fitting coarse centroids (k={coarse_k}) to score outliers"
    )
    centroids = fit_coarse_centroids(lookup, event_ids, k=coarse_k, seed=seed)
    distances = score_nearest_centroid_distance(lookup, event_ids, centroids)

    fit_ids, outlier_ids = select_local_fit_sample(
        event_ids,
        distances,
        strata,
        fit_max=fit_max,
        outlier_fraction=outlier_fraction,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    logger.info(
        f"[tier3] local UMAP fit on {len(fit_ids):,} / {n:,} events "
        f"({len(outlier_ids):,} guaranteed as outliers); "
        f"transform remainder in batches of {transform_batch:,}"
    )

    fit_vectors = embeddings_year_major(lookup, fit_ids)
    reducer = umap.UMAP(random_state=seed, **umap_params)
    fit_coords = reducer.fit_transform(fit_vectors)
    fit_clusters = leiden_cluster(
        fit_vectors, resolution_parameter=resolution_parameter, n_neighbors=n_neighbors
    )

    coords_by_id = {
        eid: fit_coords[j].astype(np.float32) for j, eid in enumerate(fit_ids)
    }
    clusters_by_id = {
        eid: int(fit_clusters[j]) for j, eid in enumerate(fit_ids)
    }

    fit_set = set(fit_ids)
    rest_ids = [eid for eid in event_ids if eid not in fit_set]

    for start in range(0, len(rest_ids), transform_batch):
        batch_ids = rest_ids[start:start + transform_batch]
        batch_vectors = embeddings_year_major(lookup, batch_ids)

        batch_coords = reducer.transform(batch_vectors)
        batch_clusters = assign_labels_by_nearest_fit(
            fit_vectors, fit_clusters, batch_vectors
        )

        for j, eid in enumerate(batch_ids):
            coords_by_id[eid] = batch_coords[j].astype(np.float32)
            clusters_by_id[eid] = int(batch_clusters[j])

        del batch_vectors

    del fit_vectors

    local_coords = np.asarray(
        [coords_by_id[eid] for eid in event_ids], dtype=np.float32
    )
    clusters = np.asarray(
        [clusters_by_id[eid] for eid in event_ids], dtype=np.int32
    )

    cluster_centroid_vectors = compute_cluster_centroid_vectors_streaming(
        lookup, event_ids, clusters
    )

    return {
        "event_ids": event_ids,
        "local_coords": local_coords,
        "clusters": clusters,
        "cluster_centroid_vectors": cluster_centroid_vectors,
        "fit_info": {
            "n": n,
            "fit_n": len(fit_ids),
            "outlier_n": len(outlier_ids),
            "sampled": True,
        },
    }