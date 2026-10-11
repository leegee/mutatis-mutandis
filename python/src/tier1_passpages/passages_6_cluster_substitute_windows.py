"""
passages_6_cluster_substitute_windows.py

Cluster the text windows written by analyse_substitute_collocates.py
(<prefix>_windows.csv) to find sense-like groupings of the seed and its
substitutes, and show which terms occupy which cluster.

Method
  1. Replace the marked hit word ([[word]]) with [MASK] and embed the window
     with MacBERTh. The vector is the (mean of the last few layers') hidden
     state at the masked position. Because the target is masked, windows of
     different words (resist / withstand / oppose ...) share one space and
     are directly comparable.
  2. Reduce with UMAP (cosine metric) to a moderate number of dimensions.
  3. Cluster with HDBSCAN, which needs no preset number of clusters and
     labels windows that fit no cluster as noise (-1).
  4. Re-run UMAP + HDBSCAN across seeds and n_neighbors settings and report
     agreement (adjusted Rand index) with the baseline clustering.

Run from the repository root:
    python src/tier1/cluster_substitute_windows.py \
        --input out/resist_matched_profiles_windows.csv \
        --random-seed 42

Note: the windows CSV inherits whatever --forbid / --require filters were used
in the analyse_substitute_collocates.py run that produced it. To see senses
that those filters remove (e.g. spiritual uses of 'resist'), re-run that script
without --forbid first.

Requires: numpy, torch, transformers, umap-learn, scikit-learn (>=1.3 for
HDBSCAN). matplotlib is optional (for the --plot PNG).

Outputs (default prefix = input stem without '_windows' + '_clusters'):
    <prefix>_windows.csv     input rows + cluster, cluster_prob, umap_x, umap_y
    <prefix>_summary.csv     one row per cluster: size, docs, years,
                             distinctive collocates, per-term counts
    <prefix>_terms.csv       cluster x term counts and shares
    <prefix>_umap.png        (with --plot)
    <prefix>_embeddings.npz  embedding cache, reused if the input is unchanged
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lib.corpus_logging import logger


MASK_RE = re.compile(r"\[\[(.+?)\]\]")


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
def read_windows(path: Path, terms: set[str] | None, max_per_term: int,
                 rng: random.Random):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for col in ("term", "passage", "collocates_present"):
        if col not in rows[0]:
            raise ValueError(f"{path} has no {col!r} column; is it a "
                             f"_windows.csv from analyse_substitute_collocates.py?")

    if terms:
        rows = [r for r in rows if r["term"] in terms]
        if not rows:
            raise ValueError("--terms matched no rows")

    order: list[str] = []
    by_term: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["term"] not in by_term:
            order.append(r["term"])
        by_term[r["term"]].append(r)

    out = []
    for term in order:
        rs = by_term[term]
        if max_per_term and len(rs) > max_per_term:
            rs = rng.sample(rs, max_per_term)
        out.extend(rs)

    keep = [r for r in out if len(MASK_RE.findall(r["passage"])) == 1]
    if len(keep) < len(out):
        logger.info(f"WARNING: dropped {len(out) - len(keep)} rows without "
                    f"exactly one [[marked]] hit word")
    return keep, order


# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------
def embed_masked(passages, model_name, batch_size, max_length, layers, device):
    """Return (X, valid): hidden state at the [MASK] position, L2-normalised."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device).eval()

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    texts = [MASK_RE.sub(mask_token, p) for p in passages]

    vecs = []
    valid = np.zeros(len(texts), dtype=bool)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            out = model(**enc)
            hs = torch.stack([out.hidden_states[l] for l in layers]).mean(0)
            for i in range(len(batch)):
                pos = (enc["input_ids"][i] == mask_id).nonzero(as_tuple=True)[0]
                if len(pos) != 1:
                    vecs.append(np.zeros(hs.shape[-1], dtype=np.float32))
                    continue
                vecs.append(hs[i, pos[0]].float().cpu().numpy())
                valid[start + i] = True
            if (start // batch_size) % 10 == 0:
                logger.info(f"  embedded {min(start + batch_size, len(texts))}"
                            f"/{len(texts)}")

    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    return X, valid


def get_embeddings(rows, cache_path: Path, args):
    passages = [r["passage"] for r in rows]
    digest = hashlib.md5(
        (args.model + "|" + args.layers + "|" + "\n".join(passages)).encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if str(z["digest"]) == digest:
            logger.info(f"Loaded cached embeddings: {cache_path}")
            return z["X"], z["valid"]
        logger.info("Embedding cache does not match input; recomputing.")
    layers = [int(x) for x in args.layers.split(",")]
    X, valid = embed_masked(passages, args.model, args.batch_size,
                            args.max_length, layers, args.device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, valid=valid, digest=np.array(digest))
    return X, valid


# ----------------------------------------------------------------------
# Reduction and clustering
# ----------------------------------------------------------------------
def run_umap(X, n_components, n_neighbors, min_dist, seed):
    import umap
    n_neighbors = max(2, min(n_neighbors, len(X) - 1))
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        min_dist=min_dist, metric="cosine", random_state=seed,
    )
    return reducer.fit_transform(X)


def run_hdbscan(Y, min_cluster_size, min_samples):
    from sklearn.cluster import HDBSCAN
    h = HDBSCAN(min_cluster_size=min_cluster_size,
                min_samples=min_samples or None)
    labels = h.fit_predict(Y)
    return labels, h.probabilities_


def stability(X, base_labels, args):
    from sklearn.metrics import adjusted_rand_score
    neighbors = [int(x) for x in args.stability_neighbors.split(",") if x.strip()]
    results = []
    for nn in neighbors:
        for s in range(args.stability_seeds):
            seed = args.random_seed + 1 + s
            Y = run_umap(X, args.umap_dim, nn, args.min_dist, seed)
            labels, _ = run_hdbscan(Y, args.min_cluster_size, args.min_samples)
            both = (base_labels >= 0) & (labels >= 0)
            ari = (adjusted_rand_score(base_labels[both], labels[both])
                   if both.sum() >= 2 and len(set(labels[both])) > 0 else float("nan"))
            results.append({
                "n_neighbors": nn, "seed": seed, "ari": ari,
                "n_clusters": len(set(labels) - {-1}),
                "noise_frac": float((labels == -1).mean()),
            })
    return results


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------
def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def distinctive_collocates(sets, labels, cluster, top=6, min_count=3):
    idx_in = [i for i, l in enumerate(labels) if l == cluster]
    n_in = len(idx_in)
    n_out = len(labels) - n_in
    total = Counter()
    for s in sets:
        total.update(s)
    cin = Counter()
    for i in idx_in:
        cin.update(sets[i])
    scored = []
    for w, a in cin.items():
        if a < min_count:
            continue
        c = total[w] - a
        odds_in = (a + 0.5) / (n_in - a + 0.5)
        odds_out = (c + 0.5) / (n_out - c + 0.5)
        scored.append((math.log(odds_in / odds_out), w, a))
    scored.sort(reverse=True)
    return [(w, a) for _, w, a in scored[:top]]


def trim_passage(passage, radius=18):
    toks = passage.split()
    centre = next((i for i, t in enumerate(toks) if t.startswith("[[")), 0)
    lo, hi = max(0, centre - radius), centre + radius + 1
    return ("... " if lo else "") + " ".join(toks[lo:hi]) + (" ..." if hi < len(toks) else "")


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_umap(path: Path, xy, labels, term_names, order):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed; skipping plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")

    ax = axes[0]
    noise = labels == -1
    ax.scatter(xy[noise, 0], xy[noise, 1], s=8, c="#bbbbbb", label="noise")
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=10, color=cmap(c % 10), label=f"cluster {c}")
    ax.set_title("By cluster")
    ax.legend(fontsize=7, markerscale=2)

    ax = axes[1]
    for i, term in enumerate(order):
        m = term_names == term
        ax.scatter(xy[m, 0], xy[m, 1], s=10, color=cmap(i % 10), label=term, alpha=0.7)
    ax.set_title("By term")
    ax.legend(fontsize=7, markerscale=2)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Cluster substitute/seed text windows by masked embedding."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="<prefix>_windows.csv from analyse_substitute_collocates.py")
    p.add_argument("--output-prefix", type=Path, default=None)
    p.add_argument("--seed", default=None,
                   help="Label of the seed term (default: first term in the CSV).")
    p.add_argument("--terms", default="",
                   help="Optional comma-separated term labels to keep.")
    p.add_argument("--max-per-term", type=int, default=0,
                   help="Randomly cap windows per term (0 = all).")
    p.add_argument("--model", default="emanjavacas/MacBERTh",
                   help="HuggingFace model name or local path.")
    p.add_argument("--layers", default="-4,-3,-2,-1",
                   help="Hidden layers averaged at the mask position.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--umap-dim", type=int, default=5,
                   help="UMAP dimensions used for clustering.")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.0)
    p.add_argument("--min-cluster-size", type=int, default=15)
    p.add_argument("--min-samples", type=int, default=0,
                   help="HDBSCAN min_samples (0 = min-cluster-size).")
    p.add_argument("--stability-seeds", type=int, default=3,
                   help="Seeds per n_neighbors setting in the stability check "
                        "(0 = skip).")
    p.add_argument("--stability-neighbors", default="10,15,30")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--log-examples", type=int, default=2,
                   help="Example passages logged per cluster.")
    p.add_argument("--plot", action="store_true", help="Write a UMAP PNG.")
    args = p.parse_args()

    rng = random.Random(args.random_seed)
    np.random.seed(args.random_seed)

    terms = {t.strip() for t in args.terms.split(",") if t.strip()} or None
    rows, order = read_windows(args.input, terms, args.max_per_term, rng)
    seed = args.seed or order[0]
    if seed not in order:
        p.error(f"--seed {seed!r} not found; terms are: {order}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Terms: {', '.join(order)} (seed: {seed})")
    logger.info(f"Windows: {len(rows)}")

    base = args.input.stem
    if base.endswith("_windows"):
        base = base[: -len("_windows")]
    prefix = args.output_prefix or args.input.with_name(base + "_clusters")
    prefix = Path(prefix)

    # 1. Embed
    X, valid = get_embeddings(rows, Path(str(prefix) + "_embeddings.npz"), args)
    if not valid.all():
        logger.info(f"WARNING: dropping {int((~valid).sum())} windows whose "
                    f"[MASK] was lost (truncation or tokenisation)")
        rows = [r for r, v in zip(rows, valid) if v]
        X = X[valid]
    n = len(rows)
    if n < max(args.min_cluster_size * 2, 10):
        raise SystemExit(f"Only {n} windows; too few to cluster.")

    term_names = np.array([r["term"] for r in rows])
    sets = [set(filter(None, r["collocates_present"].split(";"))) for r in rows]

    # 2-3. UMAP + HDBSCAN
    logger.info(f"UMAP ({args.umap_dim}d, n_neighbors={args.n_neighbors}) + "
                f"HDBSCAN (min_cluster_size={args.min_cluster_size})")
    Y = run_umap(X, args.umap_dim, args.n_neighbors, args.min_dist, args.random_seed)
    labels, probs = run_hdbscan(Y, args.min_cluster_size, args.min_samples)
    clusters = sorted(set(labels) - {-1})
    logger.info(f"Clusters: {len(clusters)}; noise: {int((labels == -1).sum())}"
                f"/{n} ({(labels == -1).mean():.0%})")
    if not clusters:
        logger.info("WARNING: HDBSCAN found no clusters. Try a smaller "
                    "--min-cluster-size or a larger --n-neighbors.")

    xy = run_umap(X, 2, args.n_neighbors, args.min_dist, args.random_seed)

    # 4. Stability
    if args.stability_seeds > 0 and clusters:
        logger.info("Stability check (ARI against baseline, non-noise windows):")
        res = stability(X, labels, args)
        for r in res:
            logger.info(f"  n_neighbors={r['n_neighbors']:<3} seed={r['seed']:<4} "
                        f"ARI={r['ari']:.3f}  clusters={r['n_clusters']}  "
                        f"noise={r['noise_frac']:.0%}")
        aris = [r["ari"] for r in res if not math.isnan(r["ari"])]
        if aris:
            logger.info(f"  mean ARI {statistics.mean(aris):.3f}, "
                        f"min {min(aris):.3f}")

    # 5. Outputs
    out_rows = []
    for r, lab, pr, (x, y) in zip(rows, labels, probs, xy):
        o = dict(r)
        o.update({"cluster": int(lab), "cluster_prob": f"{pr:.4f}",
                  "umap_x": f"{x:.4f}", "umap_y": f"{y:.4f}"})
        out_rows.append(o)
    base_fields = list(rows[0].keys())
    write_csv(Path(str(prefix) + "_windows.csv"), out_rows,
              base_fields + ["cluster", "cluster_prob", "umap_x", "umap_y"])

    term_totals = Counter(term_names)
    summary_rows, term_rows = [], []
    dist = {t: np.zeros(len(clusters)) for t in order}
    for ci, c in enumerate(clusters + [-1]):
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            continue
        name = "noise" if c == -1 else c
        counts = Counter(term_names[i] for i in idx)
        docs = {rows[i]["doc_id"] for i in idx}
        years = [to_int(rows[i].get("year")) for i in idx]
        years = [y for y in years if y is not None]
        dc = ([] if c == -1 else distinctive_collocates(sets, labels, c))
        summary_rows.append({
            "cluster": name,
            "n_windows": len(idx),
            "n_docs": len(docs),
            "year_median": int(statistics.median(years)) if years else "",
            "distinctive_collocates": ";".join(f"{w}:{a}" for w, a in dc),
            "term_counts": ";".join(f"{t}:{counts[t]}" for t in order if counts[t]),
        })
        for t in order:
            term_rows.append({
                "cluster": name, "term": t, "n_windows": counts[t],
                "share_of_term": f"{counts[t] / term_totals[t]:.4f}",
                "share_of_cluster": f"{counts[t] / len(idx):.4f}",
            })
            if c != -1:
                dist[t][ci] = counts[t]
    write_csv(Path(str(prefix) + "_summary.csv"), summary_rows,
              ["cluster", "n_windows", "n_docs", "year_median",
               "distinctive_collocates", "term_counts"])
    write_csv(Path(str(prefix) + "_terms.csv"), term_rows,
              ["cluster", "term", "n_windows", "share_of_term",
               "share_of_cluster"])
    if args.plot:
        plot_umap(Path(str(prefix) + "_umap.png"), xy, labels, term_names, order)

    # Console report
    for c in clusters:
        idx = np.array([i for i, l in enumerate(labels) if l == c])
        counts = Counter(term_names[i] for i in idx)
        dc = distinctive_collocates(sets, labels, c)
        logger.info(f"\nCluster {c}: {len(idx)} windows, "
                    f"{len({rows[i]['doc_id'] for i in idx})} docs")
        logger.info("  terms: " + ", ".join(f"{t} {counts[t]}" for t in order if counts[t]))
        if dc:
            logger.info("  distinctive collocates: " +
                        ", ".join(f"{w}({a})" for w, a in dc))
        centroid = Y[idx].mean(axis=0)
        nearest = idx[np.argsort(np.linalg.norm(Y[idx] - centroid, axis=1))]
        for i in nearest[:args.log_examples]:
            r = rows[i]
            logger.info(f"  [{r.get('year', '')}] {r['term']}: "
                        f"{trim_passage(r['passage'])}")

    if clusters:
        logger.info("\nCluster-distribution similarity to seed "
                    "(cosine of term counts across clusters, noise excluded):")
        sims = []
        for t in order:
            if t == seed:
                continue
            sims.append((cosine(dist[seed], dist[t]), t))
        for s, t in sorted(sims, reverse=True):
            logger.info(f"  {t:<26} {s:.3f}  "
                        f"({int(dist[t].sum())} clustered windows)")

    logger.info(f"\nSaved: {prefix}_windows.csv")
    logger.info(f"Saved: {prefix}_summary.csv")
    logger.info(f"Saved: {prefix}_terms.csv")
    if args.plot:
        logger.info(f"Saved: {prefix}_umap.png")


if __name__ == "__main__":
    main()
