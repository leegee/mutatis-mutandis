"""
Compare substitute and seed collocate profiles using an existing
passage_mask_fillers.py output CSV. This mode does not load MacBERTh.

Run from the repository root:
    python src/tier1/analyse_substitute_collocates.py \
        --input out/resist_matched.csv \
        --seed-forms resist,resiste \
        --forbid god,lord,sathan,devil,christ,faith,temptation,sinne \
        --top-substitutes 5 --top-collocates 20 \
        --random-seed 42

The input CSV supplies the candidate substitutes and collocates. Occurrence
windows are then retrieved from PostgreSQL so that each term's profile is
measured against the same source corpus and period.

Sampling is independent of the original CSV run; pass --random-seed for
reproducible samples. Composition statistics (documents, year distribution)
are reported so that differences in profile similarity can be interpreted
in light of sample composition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Permit importing the existing script when this file is run from the repo root.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from passage_mask_fillers import SCOPES, fetch_windows, stem_key


def csv_words(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def split_forms(value: str) -> list[str]:
    """CSV fillers use slash-separated surface variants, e.g. prevent/preuent."""
    return [
        part.strip().lower()
        for part in value.split("/")
        if part.strip() and part.strip().isalpha()
    ]


def read_candidates(path: Path, top_substitutes: int, top_collocates: int):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    substitute_rows = [
        r for r in rows
        if r.get("kind") == "substitute" and r.get("filler", "").strip()
    ]
    collocate_rows = [
        r for r in rows
        if r.get("kind") == "collocate" and r.get("filler", "").strip()
    ]
    if not substitute_rows or not collocate_rows:
        raise ValueError(
            "Input must contain both kind=substitute and kind=collocate rows."
        )

    # Rank is stored per result family. Keep the first seed/period/scope group.
    key = (
        substitute_rows[0].get("seed"),
        substitute_rows[0].get("period"),
        substitute_rows[0].get("scope"),
    )
    substitute_rows = [
        r for r in substitute_rows
        if (r.get("seed"), r.get("period"), r.get("scope")) == key
    ]
    collocate_rows = [
        r for r in collocate_rows
        if (r.get("seed"), r.get("period"), r.get("scope")) == key
    ]

    substitute_rows.sort(key=lambda r: int(r.get("rank") or 999999))
    collocate_rows.sort(key=lambda r: int(r.get("rank") or 999999))

    substitutes = []
    seen = set()
    labels_seen = set()
    for row in substitute_rows:
        forms = split_forms(row["filler"])
        forms = [f for f in forms if f not in seen]
        if not forms:
            continue
        label = row["filler"]
        if label in labels_seen:
            continue
        seen.update(forms)
        labels_seen.add(label)
        substitutes.append({
            "label": label,
            "forms": forms,
            "rank": int(row.get("rank") or 0),
            "score": row.get("score", ""),
        })
        if len(substitutes) >= top_substitutes:
            break

    collocates = []
    seen_keys = set()
    for row in collocate_rows:
        word = row["filler"].strip().lower()
        k = stem_key(word)
        if not word.isalpha() or k in seen_keys:
            continue
        seen_keys.add(k)
        collocates.append({
            "word": word,
            "key": k,
            "rank": int(row.get("rank") or 0),
            "score": row.get("score", ""),
        })
        if len(collocates) >= top_collocates:
            break

    seed = substitute_rows[0].get("seed", "resist").strip().lower()
    period = substitute_rows[0].get("period", "")
    scope = substitute_rows[0].get("scope", "all")
    if "-" not in period:
        raise ValueError(f"Expected period like 1550-1700, got {period!r}")
    lo, hi = (int(x) for x in period.split("-", 1))
    if scope not in SCOPES:
        raise ValueError(f"CSV scope {scope!r} is not a supported database scope")

    return seed, lo, hi, scope, substitutes, collocates


def _stable_order_expr(seed: int | None) -> str:
    """Return a deterministic ORDER BY expression when a seed is supplied."""
    if seed is None:
        return "random()"
    # md5 gives a stable pseudo-random order that does not depend on
    # PostgreSQL's session random state.
    return f"md5(t.doc_id::text || ':' || t.token_idx::text || ':{seed}')"


def fetch_term_hits(conn, scope, forms, lo, hi, per_doc, limit, random_seed):
    """Sample up to per_doc occurrences per document, then cap total rows."""
    sc = SCOPES[scope]
    order_expr = _stable_order_expr(random_seed)
    # The window function also needs a stable order when seeded.
    rn_order = order_expr if random_seed is not None else "random()"

    query = f"""
        WITH hits AS (
            SELECT t.doc_id, t.token_idx, lower(t.token) AS surface,
                   d.pub_year,
                   row_number() OVER (
                       PARTITION BY t.doc_id ORDER BY {rn_order}
                   ) AS rn
            FROM tokens t
            JOIN {sc['table']} d ON {sc['join']}
            WHERE lower(t.token) = ANY(%(forms)s)
              AND d.pub_year BETWEEN %(lo)s AND %(hi)s
              {sc['where']}
        )
        SELECT doc_id, token_idx, surface, pub_year
        FROM hits
        WHERE rn <= %(per_doc)s
        ORDER BY {order_expr.replace('t.', '')}
        LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(query, {
            "forms": sorted(set(forms)),
            "lo": lo,
            "hi": hi,
            "per_doc": per_doc,
            "limit": limit,
        })
        return [
            {"doc_id": r[0], "token_idx": r[1], "surface": r[2], "year": r[3]}
            for r in cur.fetchall()
        ]


def window_passes(words, centre_idx, require, forbid):
    context_words = {
        word.lower()
        for i, word in enumerate(words)
        if i != centre_idx
    }
    if require and not (context_words & require):
        return False
    if forbid and (context_words & forbid):
        return False
    return True


def presence_profile(windows, collocates, excluded_keys):
    """
    Count each collocate at most once per occurrence window.

    Matching is performed after stem_key normalisation. Consequently a
    collocate label such as "power" may be credited for any surface form
    that reduces to the same key (e.g. powers). The observed surfaces are
    returned so the mapping can be audited.
    """
    counts = Counter()
    surfaces_by_key = defaultdict(Counter)
    usable = 0
    by_key = {c["key"]: c for c in collocates}
    for hit, idxs, words in windows:
        try:
            centre = idxs.index(hit[1])
        except ValueError:
            continue
        keys = set()
        for i, word in enumerate(words):
            if i == centre:
                continue
            w = word.lower()
            if not w.isalpha() or len(w) < 2:
                continue
            k = stem_key(w)
            if k in excluded_keys:
                continue
            if k in by_key:
                keys.add(k)
                surfaces_by_key[k][w] += 1
        usable += 1
        counts.update(keys)
    return usable, counts, surfaces_by_key


def decade_histogram(years: list[int]) -> str:
    if not years:
        return ""
    bins = Counter((y // 10) * 10 for y in years)
    return ";".join(f"{d}-{d+9}:{bins[d]}" for d in sorted(bins))


def retrieve_profile(conn, scope, forms, lo, hi, context, per_doc, limit,
                     require, forbid, collocates, excluded_keys, random_seed):
    hits = fetch_term_hits(
        conn, scope, forms, lo, hi,
        per_doc=per_doc, limit=limit, random_seed=random_seed,
    )
    if not hits:
        return {
            "n": 0,
            "counts": Counter(),
            "surfaces": Counter(),
            "collocate_surfaces": {},
            "doc_ids": set(),
            "years": [],
        }

    hit_by_key = {(h["doc_id"], h["token_idx"]): h for h in hits}
    hit_pairs = list(hit_by_key.keys())
    windows = fetch_windows(conn, hit_pairs, context=context)

    filtered = []
    surface_counts = Counter()
    doc_ids = set()
    years = []
    for hit, idxs, words in windows:
        try:
            centre = idxs.index(hit[1])
        except ValueError:
            continue
        if not window_passes(words, centre, require, forbid):
            continue
        filtered.append((hit, idxs, words))
        meta = hit_by_key.get((hit[0], hit[1]))
        if meta is not None:
            surface_counts[meta["surface"]] += 1
            doc_ids.add(meta["doc_id"])
            years.append(meta["year"])

    n, counts, collocate_surfaces = presence_profile(
        filtered, collocates, excluded_keys
    )
    return {
        "n": n,
        "counts": counts,
        "surfaces": surface_counts,
        "collocate_surfaces": collocate_surfaces,
        "doc_ids": doc_ids,
        "years": years,
    }


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description="Compare substitute collocate profiles from an existing CSV."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Existing passage_mask_fillers output CSV.")
    p.add_argument("--output-prefix", type=Path, default=None,
                   help="Output path prefix; defaults to input stem + '_profiles'.")
    p.add_argument("--seed-forms", default="resist,resiste",
                   help="Comma-separated seed forms used to retrieve the seed profile.")
    p.add_argument("--top-substitutes", type=int, default=5)
    p.add_argument("--top-collocates", type=int, default=20)
    p.add_argument("--context", type=int, default=40,
                   help="Tokens either side of each occurrence; match original run.")
    p.add_argument("--max-per-doc", type=int, default=10,
                   help="Maximum sampled occurrences of each term per document.")
    p.add_argument("--max-occurrences", type=int, default=20000,
                   help="Total occurrence cap per term.")
    p.add_argument("--require", default="",
                   help="Optional comma-separated words required in the context.")
    p.add_argument("--forbid", default="",
                   help="Words forbidden in the context; mirror the original run.")
    p.add_argument("--exclude-forms", default="resistance",
                   help="Comma-separated forms excluded from collocate profiles.")
    p.add_argument("--min-windows", type=int, default=30,
                   help="Minimum usable windows before a cosine is reported.")
    p.add_argument("--random-seed", type=int, default=None,
                   help="If set, make sampling deterministic across runs.")
    args = p.parse_args()

    if args.top_substitutes < 1 or args.top_collocates < 1:
        p.error("--top-substitutes and --top-collocates must be positive")
    if args.context < 1 or args.max_per_doc < 1 or args.max_occurrences < 1:
        p.error("--context, --max-per-doc, and --max-occurrences must be positive")

    if args.random_seed is not None:
        random.seed(args.random_seed)

    seed, lo, hi, scope, substitutes, collocates = read_candidates(
        args.input, args.top_substitutes, args.top_collocates
    )
    seed_forms = csv_words(args.seed_forms)
    require, forbid = csv_words(args.require), csv_words(args.forbid)
    excluded_keys = {stem_key(x) for x in csv_words(args.exclude_forms)}
    excluded_keys.update(stem_key(x) for x in seed_forms)

    # Reject any substitute that overlaps the seed forms.
    cleaned = []
    for s in substitutes:
        overlap = set(s["forms"]) & seed_forms
        if overlap:
            print(f"WARNING: dropping substitute {s['label']!r} "
                  f"(overlaps seed forms {sorted(overlap)})")
            continue
        cleaned.append(s)
    substitutes = cleaned

    print(f"CSV: {args.input}")
    print(f"Corpus scope: {scope}; years: {lo}-{hi}")
    print(f"Seed: {seed} ({', '.join(sorted(seed_forms))})")
    print("Substitutes: " + ", ".join(s["label"] for s in substitutes))
    print("Collocates: " + ", ".join(c["word"] for c in collocates))
    if forbid:
        print("Forbidding: " + ", ".join(sorted(forbid)))
    if args.random_seed is not None:
        print(f"Random seed: {args.random_seed}")

    from lib.corpus_db import get_connection
    conn = get_connection(application_name="substitute-collocate-profiles")
    try:
        targets = [{
            "label": seed,
            "forms": sorted(seed_forms),
            "rank": 0,
            "score": "",
        }] + substitutes

        profiles = {}
        matrix_rows = []
        detail_rows = []
        surface_rows = []

        for target in targets:
            result = retrieve_profile(
                conn=conn,
                scope=scope,
                forms=target["forms"],
                lo=lo,
                hi=hi,
                context=args.context,
                per_doc=args.max_per_doc,
                limit=args.max_occurrences,
                require=require,
                forbid=forbid,
                collocates=collocates,
                excluded_keys=excluded_keys,
                random_seed=args.random_seed,
            )
            n = result["n"]
            counts = result["counts"]
            rates = [
                counts.get(c["key"], 0) / n if n else 0.0
                for c in collocates
            ]
            years = result["years"]
            profiles[target["label"]] = {
                "n": n,
                "counts": counts,
                "rates": rates,
                "surfaces": result["surfaces"],
                "collocate_surfaces": result["collocate_surfaces"],
                "doc_ids": result["doc_ids"],
                "years": years,
                "forms": target["forms"],
                "score": target["score"],
                "rank": target["rank"],
            }
            if n < args.min_windows:
                print(f"WARNING: {target['label']}: only {n} usable windows "
                      f"(cosine will be blank)")

            for c, rate in zip(collocates, rates):
                matrix_rows.append({
                    "term": target["label"],
                    "forms": "/".join(target["forms"]),
                    "n_windows": n,
                    "collocate": c["word"],
                    "collocate_rank_in_seed": c["rank"],
                    "cooccurrence_windows": counts.get(c["key"], 0),
                    "window_rate": f"{rate:.6f}",
                    "collocate_seed_score": c["score"],
                })

            # Audit trail for stem_key matches.
            for c in collocates:
                obs = result["collocate_surfaces"].get(c["key"], Counter())
                if not obs:
                    continue
                for surface, cnt in obs.most_common():
                    surface_rows.append({
                        "term": target["label"],
                        "collocate_label": c["word"],
                        "collocate_key": c["key"],
                        "observed_surface": surface,
                        "count": cnt,
                    })

        seed_profile = profiles[seed]["rates"]
        seed_n = profiles[seed]["n"]
        seed_docs = profiles[seed]["doc_ids"]

        for target in substitutes:
            profile = profiles[target["label"]]
            rates = profile["rates"]
            for c, seed_rate, target_rate in zip(
                collocates, seed_profile, rates
            ):
                seed_count = profiles[seed]["counts"].get(c["key"], 0)
                target_count = profile["counts"].get(c["key"], 0)
                seed_smoothed = (seed_count + 0.5) / (seed_n + 1) if seed_n else 0
                target_smoothed = (
                    (target_count + 0.5) / (profile["n"] + 1)
                    if profile["n"] else 0
                )
                detail_rows.append({
                    "substitute": target["label"],
                    "substitute_rank": target["rank"],
                    "substitute_mlm_score": target["score"],
                    "seed": seed,
                    "collocate": c["word"],
                    "seed_windows": seed_n,
                    "seed_cooccurrence_windows": seed_count,
                    "seed_window_rate": f"{seed_rate:.6f}",
                    "substitute_windows": profile["n"],
                    "substitute_cooccurrence_windows": target_count,
                    "substitute_window_rate": f"{target_rate:.6f}",
                    "rate_difference": f"{target_rate - seed_rate:.6f}",
                    "smoothed_rate_ratio": (
                        f"{target_smoothed / seed_smoothed:.6f}"
                        if seed_smoothed else ""
                    ),
                })

        # One clean summary row per term, with composition statistics.
        summary_rows = []
        for target in targets:
            profile = profiles[target["label"]]
            is_seed = target["label"] == seed
            years = profile["years"]
            n = profile["n"]
            reliable = n >= args.min_windows

            if is_seed:
                cosine_val = "1.000000"
                doc_overlap = ""
            else:
                if reliable and seed_n >= args.min_windows:
                    cosine_val = f"{cosine(seed_profile, profile['rates']):.6f}"
                else:
                    cosine_val = ""
                # Simple document-overlap diagnostic.
                overlap = len(profile["doc_ids"] & seed_docs)
                union = len(profile["doc_ids"] | seed_docs)
                doc_overlap = (
                    f"{overlap}/{len(profile['doc_ids'])} "
                    f"(Jaccard {overlap / union:.3f})" if union else ""
                )

            summary_rows.append({
                "term": target["label"],
                "forms": "/".join(target["forms"]),
                "n_windows": n,
                "n_docs": len(profile["doc_ids"]),
                "year_min": min(years) if years else "",
                "year_median": (
                    int(statistics.median(years)) if years else ""
                ),
                "year_max": max(years) if years else "",
                "year_mean": f"{statistics.mean(years):.1f}" if years else "",
                "year_decades": decade_histogram(years),
                "doc_overlap_with_seed": doc_overlap,
                "most_common_surface_forms": ";".join(
                    f"{w}:{c}" for w, c in profile["surfaces"].most_common(8)
                ),
                "cosine_to_seed_profile": cosine_val,
                "mlm_rank": 0 if is_seed else target["rank"],
                "mlm_score": "" if is_seed else target["score"],
            })

        prefix = args.output_prefix or args.input.with_name(
            args.input.stem + "_profiles"
        )
        write_csv(
            Path(str(prefix) + "_matrix.csv"),
            matrix_rows,
            ["term", "forms", "n_windows", "collocate",
             "collocate_rank_in_seed", "cooccurrence_windows", "window_rate",
             "collocate_seed_score"],
        )
        write_csv(
            Path(str(prefix) + "_detail.csv"),
            detail_rows,
            ["substitute", "substitute_rank", "substitute_mlm_score", "seed",
             "collocate", "seed_windows", "seed_cooccurrence_windows",
             "seed_window_rate", "substitute_windows",
             "substitute_cooccurrence_windows", "substitute_window_rate",
             "rate_difference", "smoothed_rate_ratio"],
        )
        write_csv(
            Path(str(prefix) + "_summary.csv"),
            summary_rows,
            ["term", "forms", "n_windows", "n_docs",
             "year_min", "year_median", "year_max", "year_mean",
             "year_decades", "doc_overlap_with_seed",
             "most_common_surface_forms", "cosine_to_seed_profile",
             "mlm_rank", "mlm_score"],
        )
        write_csv(
            Path(str(prefix) + "_collocate_surfaces.csv"),
            surface_rows,
            ["term", "collocate_label", "collocate_key",
             "observed_surface", "count"],
        )

        print(f"\nSeed profile windows: {seed_n} "
              f"across {len(seed_docs)} documents")
        print("Profile similarity to seed (selected-collocate cosine):")
        for target in sorted(
            substitutes,
            key=lambda t: (
                cosine(seed_profile, profiles[t["label"]]["rates"])
                if profiles[t["label"]]["n"] >= args.min_windows else -1
            ),
            reverse=True,
        ):
            prof = profiles[target["label"]]
            if prof["n"] < args.min_windows:
                score = "n/a"
            else:
                score = f"{cosine(seed_profile, prof['rates']):.3f}"
            print(
                f"  {target['label']:<26} {score:>6} "
                f"({prof['n']} windows, {len(prof['doc_ids'])} docs)"
            )
        print(f"\nSaved: {prefix}_matrix.csv")
        print(f"Saved: {prefix}_detail.csv")
        print(f"Saved: {prefix}_summary.csv")
        print(f"Saved: {prefix}_collocate_surfaces.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
