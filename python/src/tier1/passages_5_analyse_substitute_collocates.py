"""
tier1/passages_5_analyse_substitute_collocates.py

Compare substitute and seed collocate profiles using an existing
passage_mask_fillers.py output CSV. This mode does not load MacBERTh.

Run from the repository root:
    python src/tier1/analyse_substitute_collocates.py \
        --input out/resist_matched.csv \
        --seed-forms resist,resiste \
        --forbid god,lord,sathan,devil,christ,faith,temptation,sinne \
        --top-substitutes 5 --top-collocates 20 \
        --random-seed 42 \
        --match-documents \
        --match-decades

The input CSV supplies the candidate substitutes and collocates. Occurrence
windows are then retrieved from PostgreSQL so that each term's profile is
measured against the same source corpus and period.

--match-documents   restricts substitutes to documents that contributed
                    usable seed windows.
--match-decades     further stratifies by decade: only decades with enough
                    seed windows are used, and substitutes are drawn from
                    the same decade (and, if matching documents, from the
                    seed documents active in that decade).

Sampling is independent of the original CSV run; pass --random-seed for
reproducible samples. Composition statistics are always reported.

Matched documents (with year and title) are written to <prefix>_documents.csv
(and <prefix>_decade_documents.csv with --match-decades) and logged per term.
The text window around each hit, with the hit word marked [[like this]], is
written to <prefix>_windows.csv (and <prefix>_decade_windows.csv).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lib.corpus_config import OUT_DIR
from lib.corpus_logging import logger
from lib.corpus_db import get_connection


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from passages_4_mask_fillers import SCOPES, fetch_windows, stem_key


def csv_words(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def split_forms(value: str) -> list[str]:
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


def _stable_order_expr(seed: int | None, table_alias: str = "t") -> str:
    if seed is None:
        return "random()"
    return (
        f"md5({table_alias}.doc_id::text || ':' || "
        f"{table_alias}.token_idx::text || ':{seed}')"
    )


def fetch_term_hits(conn, scope, forms, lo, hi, per_doc, limit,
                    random_seed, allowed_docs=None):
    sc = SCOPES[scope]
    order_expr = _stable_order_expr(random_seed, "t")
    rn_order = order_expr if random_seed is not None else "random()"

    doc_filter = ""
    params = {
        "forms": sorted(set(forms)),
        "lo": lo,
        "hi": hi,
        "per_doc": per_doc,
        "limit": limit,
    }
    if allowed_docs is not None:
        if not allowed_docs:
            return []
        doc_filter = "AND t.doc_id = ANY(%(allowed_docs)s)"
        params["allowed_docs"] = list(allowed_docs)

    final_order = (
        _stable_order_expr(random_seed, "hits").replace("hits.", "")
        if random_seed is not None
        else "random()"
    )

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
              {doc_filter}
        )
        SELECT doc_id, token_idx, surface, pub_year
        FROM hits
        WHERE rn <= %(per_doc)s
        ORDER BY {final_order}
        LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        return [
            {"doc_id": r[0], "token_idx": r[1], "surface": r[2], "year": r[3]}
            for r in cur.fetchall()
        ]


def fetch_doc_metadata(conn, scope, doc_ids, title_column="title"):
    """Return {doc_id: {"title": ..., "year": ...}} for the given documents."""
    if not doc_ids:
        return {}
    sc = SCOPES[scope]
    query = f"""
        SELECT d.doc_id, d.{title_column}, d.pub_year
        FROM {sc['table']} d
        WHERE d.doc_id = ANY(%(ids)s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, {"ids": sorted(doc_ids)})
            return {
                r[0]: {"title": r[1] or "", "year": r[2]}
                for r in cur.fetchall()
            }
    except Exception as e:
        conn.rollback()
        logger.info(f"WARNING: could not fetch document metadata ({e}); "
                    f"titles and years will be blank.")
        return {}


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


def window_collocates(words, centre, by_key, excluded_keys):
    """Return the selected collocate words present in a window (in text order)."""
    found = []
    for i, word in enumerate(words):
        if i == centre:
            continue
        w = word.lower()
        if not w.isalpha() or len(w) < 2:
            continue
        k = stem_key(w)
        if k in excluded_keys:
            continue
        if k in by_key and by_key[k]["word"] not in found:
            found.append(by_key[k]["word"])
    return found


def build_window_rows(term, windows, hit_meta, collocates, excluded_keys,
                      decade=None, limit=0):
    """One row per window: hit word marked [[like this]] in the passage."""
    by_key = {c["key"]: c for c in collocates}
    rows = []
    for hit, idxs, words in windows:
        try:
            centre = idxs.index(hit[1])
        except ValueError:
            continue
        meta = hit_meta.get((hit[0], hit[1]))
        if meta is None:
            continue
        passage = " ".join(
            f"[[{w}]]" if i == centre else w for i, w in enumerate(words)
        )
        row = {
            "term": term,
            "doc_id": meta["doc_id"],
            "year": meta["year"],
            "title": "",            # filled in once document metadata is fetched
            "token_idx": meta["token_idx"],
            "surface": meta["surface"],
            "collocates_present": ";".join(
                window_collocates(words, centre, by_key, excluded_keys)
            ),
            "passage": passage,
        }
        if decade is not None:
            row["decade"] = decade
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def retrieve_profile(conn, scope, forms, lo, hi, context, per_doc, limit,
                     require, forbid, collocates, excluded_keys,
                     random_seed, allowed_docs=None):
    hits = fetch_term_hits(
        conn, scope, forms, lo, hi,
        per_doc=per_doc, limit=limit, random_seed=random_seed,
        allowed_docs=allowed_docs,
    )
    if not hits:
        return {
            "n": 0,
            "counts": Counter(),
            "surfaces": Counter(),
            "collocate_surfaces": {},
            "doc_ids": set(),
            "years": [],
            "windows": [],          # keep raw filtered windows for decade binning
            "hit_meta": {},
            "doc_counts": Counter(),
        }

    hit_by_key = {(h["doc_id"], h["token_idx"]): h for h in hits}
    hit_pairs = list(hit_by_key.keys())
    windows = fetch_windows(conn, hit_pairs, context=context)

    filtered = []
    surface_counts = Counter()
    doc_ids = set()
    doc_counts = Counter()
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
            doc_counts[meta["doc_id"]] += 1
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
        "windows": filtered,   # list of (hit, idxs, words)
        "hit_meta": hit_by_key,
        "doc_counts": doc_counts,
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


def profile_from_windows(filtered_windows, hit_meta, collocates, excluded_keys):
    """Build a profile dict from an already-filtered list of windows."""
    surface_counts = Counter()
    doc_ids = set()
    doc_counts = Counter()
    years = []
    for hit, idxs, words in filtered_windows:
        meta = hit_meta.get((hit[0], hit[1]))
        if meta is not None:
            surface_counts[meta["surface"]] += 1
            doc_ids.add(meta["doc_id"])
            doc_counts[meta["doc_id"]] += 1
            years.append(meta["year"])
    n, counts, collocate_surfaces = presence_profile(
        filtered_windows, collocates, excluded_keys
    )
    rates = [
        counts.get(c["key"], 0) / n if n else 0.0
        for c in collocates
    ]
    return {
        "n": n,
        "counts": counts,
        "rates": rates,
        "surfaces": surface_counts,
        "collocate_surfaces": collocate_surfaces,
        "doc_ids": doc_ids,
        "doc_counts": doc_counts,
        "years": years,
    }


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
                   help="Minimum usable windows before a global cosine is reported.")
    p.add_argument("--min-windows-per-decade", type=int, default=15,
                   help="Minimum seed windows in a decade before it is used "
                        "for stratified comparison.")
    p.add_argument("--random-seed", type=int, default=None,
                   help="If set, make sampling deterministic across runs.")
    p.add_argument("--match-documents", action="store_true",
                   help="Restrict substitutes to documents that contributed "
                        "usable seed windows.")
    p.add_argument("--match-decades", action="store_true",
                   help="Stratify comparison by decade; only decades with "
                        "enough seed windows are used.")
    p.add_argument("--title-column", default="title",
                   help="Column in the documents table holding the title.")
    p.add_argument("--log-documents", type=int, default=25,
                   help="Max documents logged per term (0 = all). "
                        "The CSV always lists every document.")
    p.add_argument("--log-windows", type=int, default=3,
                   help="Example text windows logged per term (0 = none).")
    p.add_argument("--max-window-rows", type=int, default=0,
                   help="Max windows per term written to the windows CSVs "
                        "(0 = all; the full set can be large).")
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

    cleaned = []
    for s in substitutes:
        overlap = set(s["forms"]) & seed_forms
        if overlap:
            logger.info(f"WARNING: dropping substitute {s['label']!r} "
                  f"(overlaps seed forms {sorted(overlap)})")
            continue
        cleaned.append(s)
    substitutes = cleaned

    logger.info(f"CSV: {args.input}")
    logger.info(f"Corpus scope: {scope}; years: {lo}-{hi}")
    logger.info(f"Seed: {seed} ({', '.join(sorted(seed_forms))})")
    logger.info("Substitutes: " + ", ".join(s["label"] for s in substitutes))
    logger.info("Collocates: " + ", ".join(c["word"] for c in collocates))
    if forbid:
        logger.info("Forbidding: " + ", ".join(sorted(forbid)))
    if args.random_seed is not None:
        logger.info(f"Random seed: {args.random_seed}")
    if args.match_documents:
        logger.info("Document matching: ON")
    if args.match_decades:
        logger.info(f"Decade matching: ON (min {args.min_windows_per_decade} "
              f"seed windows per decade)")

    conn = get_connection(application_name="substitute-collocate-profiles")
    try:
        # ------------------------------------------------------------------
        # 1. Seed profile (global)
        # ------------------------------------------------------------------
        seed_target = {
            "label": seed,
            "forms": sorted(seed_forms),
            "rank": 0,
            "score": "",
        }
        seed_result = retrieve_profile(
            conn=conn,
            scope=scope,
            forms=seed_target["forms"],
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
            allowed_docs=None,
        )
        seed_n = seed_result["n"]
        seed_docs = seed_result["doc_ids"]
        seed_rates = [
            seed_result["counts"].get(c["key"], 0) / seed_n if seed_n else 0.0
            for c in collocates
        ]
        seed_hit_meta = seed_result["hit_meta"]
        seed_windows = seed_result["windows"]

        if seed_n < args.min_windows:
            logger.info(f"WARNING: seed has only {seed_n} usable windows")

        if args.match_documents and not seed_docs:
            logger.info("ERROR: --match-documents requested but seed produced "
                  "no usable documents. Aborting.")
            return

        allowed_for_subs_global = seed_docs if args.match_documents else None
        if args.match_documents:
            logger.info(f"Seed contributed {len(seed_docs)} documents; "
                  f"global substitute profiles restricted to this set.")

        # ------------------------------------------------------------------
        # 2. Global profiles for substitutes
        # ------------------------------------------------------------------
        targets = [seed_target] + substitutes
        profiles = {
            seed: {
                "n": seed_n,
                "counts": seed_result["counts"],
                "rates": seed_rates,
                "surfaces": seed_result["surfaces"],
                "collocate_surfaces": seed_result["collocate_surfaces"],
                "doc_ids": seed_docs,
                "doc_counts": seed_result["doc_counts"],
                "windows": seed_windows,
                "hit_meta": seed_hit_meta,
                "years": seed_result["years"],
                "forms": seed_target["forms"],
                "score": "",
                "rank": 0,
            }
        }

        matrix_rows = []
        detail_rows = []
        surface_rows = []

        # Seed rows
        for c, rate in zip(collocates, seed_rates):
            matrix_rows.append({
                "term": seed,
                "forms": "/".join(seed_target["forms"]),
                "n_windows": seed_n,
                "collocate": c["word"],
                "collocate_rank_in_seed": c["rank"],
                "cooccurrence_windows": seed_result["counts"].get(c["key"], 0),
                "window_rate": f"{rate:.6f}",
                "collocate_seed_score": c["score"],
            })
        for c in collocates:
            obs = seed_result["collocate_surfaces"].get(c["key"], Counter())
            for surface, cnt in obs.most_common():
                surface_rows.append({
                    "term": seed,
                    "collocate_label": c["word"],
                    "collocate_key": c["key"],
                    "observed_surface": surface,
                    "count": cnt,
                })

        for target in substitutes:
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
                allowed_docs=allowed_for_subs_global,
            )
            n = result["n"]
            counts = result["counts"]
            rates = [
                counts.get(c["key"], 0) / n if n else 0.0
                for c in collocates
            ]
            profiles[target["label"]] = {
                "n": n,
                "counts": counts,
                "rates": rates,
                "surfaces": result["surfaces"],
                "collocate_surfaces": result["collocate_surfaces"],
                "doc_ids": result["doc_ids"],
                "doc_counts": result["doc_counts"],
                "windows": result["windows"],
                "hit_meta": result["hit_meta"],
                "years": result["years"],
                "forms": target["forms"],
                "score": target["score"],
                "rank": target["rank"],
            }
            if n < args.min_windows:
                logger.info(f"WARNING: {target['label']}: only {n} usable windows "
                      f"(global cosine will be blank)")

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

            for c in collocates:
                obs = result["collocate_surfaces"].get(c["key"], Counter())
                for surface, cnt in obs.most_common():
                    surface_rows.append({
                        "term": target["label"],
                        "collocate_label": c["word"],
                        "collocate_key": c["key"],
                        "observed_surface": surface,
                        "count": cnt,
                    })

            for c, seed_rate, target_rate in zip(collocates, seed_rates, rates):
                seed_count = seed_result["counts"].get(c["key"], 0)
                target_count = counts.get(c["key"], 0)
                seed_smoothed = (seed_count + 0.5) / (seed_n + 1) if seed_n else 0
                target_smoothed = (target_count + 0.5) / (n + 1) if n else 0
                detail_rows.append({
                    "substitute": target["label"],
                    "substitute_rank": target["rank"],
                    "substitute_mlm_score": target["score"],
                    "seed": seed,
                    "collocate": c["word"],
                    "seed_windows": seed_n,
                    "seed_cooccurrence_windows": seed_count,
                    "seed_window_rate": f"{seed_rate:.6f}",
                    "substitute_windows": n,
                    "substitute_cooccurrence_windows": target_count,
                    "substitute_window_rate": f"{target_rate:.6f}",
                    "rate_difference": f"{target_rate - seed_rate:.6f}",
                    "smoothed_rate_ratio": (
                        f"{target_smoothed / seed_smoothed:.6f}"
                        if seed_smoothed else ""
                    ),
                })

        # ------------------------------------------------------------------
        # 3. Decade-stratified profiles (optional)
        # ------------------------------------------------------------------
        decade_summary_rows = []
        decade_matrix_rows = []
        decade_document_rows = []
        decade_window_rows = []

        if args.match_decades:
            # Bin seed windows by decade
            seed_by_decade = defaultdict(list)
            seed_docs_by_decade = defaultdict(set)
            for hit, idxs, words in seed_windows:
                meta = seed_hit_meta.get((hit[0], hit[1]))
                if meta is None:
                    continue
                decade = (meta["year"] // 10) * 10
                seed_by_decade[decade].append((hit, idxs, words))
                seed_docs_by_decade[decade].add(meta["doc_id"])

            usable_decades = sorted(
                d for d, wins in seed_by_decade.items()
                if len(wins) >= args.min_windows_per_decade
            )
            logger.info(f"\nDecade stratification: {len(usable_decades)} decades "
                  f"with ≥{args.min_windows_per_decade} seed windows: "
                  f"{usable_decades}")

            for decade in usable_decades:
                d_lo, d_hi = decade, decade + 9
                seed_wins_d = seed_by_decade[decade]
                seed_prof_d = profile_from_windows(
                    seed_wins_d, seed_hit_meta, collocates, excluded_keys
                )
                seed_rates_d = seed_prof_d["rates"]
                seed_n_d = seed_prof_d["n"]

                # Documents active for the seed in this decade
                allowed_docs_d = (
                    seed_docs_by_decade[decade]
                    if args.match_documents else None
                )

                # Seed row for this decade
                for c, rate in zip(collocates, seed_rates_d):
                    decade_matrix_rows.append({
                        "decade": f"{d_lo}-{d_hi}",
                        "term": seed,
                        "forms": "/".join(seed_target["forms"]),
                        "n_windows": seed_n_d,
                        "collocate": c["word"],
                        "cooccurrence_windows": seed_prof_d["counts"].get(c["key"], 0),
                        "window_rate": f"{rate:.6f}",
                    })

                decade_summary_rows.append({
                    "decade": f"{d_lo}-{d_hi}",
                    "term": seed,
                    "forms": "/".join(seed_target["forms"]),
                    "n_windows": seed_n_d,
                    "n_docs": len(seed_prof_d["doc_ids"]),
                    "cosine_to_seed_profile": "1.000000",
                    "mlm_rank": 0,
                    "mlm_score": "",
                })

                for doc_id, cnt in seed_prof_d["doc_counts"].items():
                    decade_document_rows.append({
                        "decade": f"{d_lo}-{d_hi}",
                        "term": seed,
                        "doc_id": doc_id,
                        "n_windows": cnt,
                        "in_seed": "yes",
                    })
                decade_window_rows.extend(build_window_rows(
                    seed, seed_wins_d, seed_hit_meta, collocates,
                    excluded_keys, decade=f"{d_lo}-{d_hi}",
                    limit=args.max_window_rows,
                ))

                for target in substitutes:
                    result_d = retrieve_profile(
                        conn=conn,
                        scope=scope,
                        forms=target["forms"],
                        lo=d_lo,
                        hi=d_hi,
                        context=args.context,
                        per_doc=args.max_per_doc,
                        limit=args.max_occurrences,
                        require=require,
                        forbid=forbid,
                        collocates=collocates,
                        excluded_keys=excluded_keys,
                        random_seed=args.random_seed,
                        allowed_docs=allowed_docs_d,
                    )
                    n_d = result_d["n"]
                    rates_d = [
                        result_d["counts"].get(c["key"], 0) / n_d if n_d else 0.0
                        for c in collocates
                    ]
                    reliable = (
                        n_d >= args.min_windows_per_decade
                        and seed_n_d >= args.min_windows_per_decade
                    )
                    cos_d = (
                        f"{cosine(seed_rates_d, rates_d):.6f}"
                        if reliable else ""
                    )

                    for c, rate in zip(collocates, rates_d):
                        decade_matrix_rows.append({
                            "decade": f"{d_lo}-{d_hi}",
                            "term": target["label"],
                            "forms": "/".join(target["forms"]),
                            "n_windows": n_d,
                            "collocate": c["word"],
                            "cooccurrence_windows": result_d["counts"].get(c["key"], 0),
                            "window_rate": f"{rate:.6f}",
                        })

                    decade_summary_rows.append({
                        "decade": f"{d_lo}-{d_hi}",
                        "term": target["label"],
                        "forms": "/".join(target["forms"]),
                        "n_windows": n_d,
                        "n_docs": len(result_d["doc_ids"]),
                        "cosine_to_seed_profile": cos_d,
                        "mlm_rank": target["rank"],
                        "mlm_score": target["score"],
                    })

                    for doc_id, cnt in result_d["doc_counts"].items():
                        decade_document_rows.append({
                            "decade": f"{d_lo}-{d_hi}",
                            "term": target["label"],
                            "doc_id": doc_id,
                            "n_windows": cnt,
                            "in_seed": (
                                "yes" if doc_id in seed_docs_by_decade[decade]
                                else "no"
                            ),
                        })
                    decade_window_rows.extend(build_window_rows(
                        target["label"], result_d["windows"],
                        result_d["hit_meta"], collocates, excluded_keys,
                        decade=f"{d_lo}-{d_hi}",
                        limit=args.max_window_rows,
                    ))

        # ------------------------------------------------------------------
        # 4. Global summary
        # ------------------------------------------------------------------
        summary_rows = []
        for target in targets:
            label = target["label"]
            profile = profiles[label]
            is_seed = label == seed
            years = profile["years"]
            n = profile["n"]
            reliable = n >= args.min_windows and seed_n >= args.min_windows

            if is_seed:
                cosine_val = "1.000000"
                doc_overlap = ""
            else:
                cosine_val = (
                    f"{cosine(seed_rates, profile['rates']):.6f}"
                    if reliable else ""
                )
                overlap = len(profile["doc_ids"] & seed_docs)
                union = len(profile["doc_ids"] | seed_docs)
                doc_overlap = (
                    f"{overlap}/{len(profile['doc_ids'])} "
                    f"(Jaccard {overlap / union:.3f})" if union else "0/0"
                )

            summary_rows.append({
                "term": label,
                "forms": "/".join(target["forms"]),
                "n_windows": n,
                "n_docs": len(profile["doc_ids"]),
                "year_min": min(years) if years else "",
                "year_median": int(statistics.median(years)) if years else "",
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
                "match_documents": "yes" if args.match_documents else "no",
                "match_decades": "yes" if args.match_decades else "no",
            })

        # ------------------------------------------------------------------
        # 5. Write outputs
        # ------------------------------------------------------------------
        prefix = args.output_prefix or args.input.with_name(args.input.stem + "_profiles")

        # ------------------------------------------------------------------
        # 5a. Matched documents (with dates and titles)
        # ------------------------------------------------------------------
        all_doc_ids = set()
        for prof in profiles.values():
            all_doc_ids |= set(prof["doc_counts"])
        for r in decade_document_rows:
            all_doc_ids.add(r["doc_id"])
        doc_meta = fetch_doc_metadata(conn, scope, all_doc_ids, args.title_column)

        def meta_for(doc_id):
            m = doc_meta.get(doc_id, {})
            return m.get("year") or "", m.get("title") or ""

        term_order = {t["label"]: i for i, t in enumerate(targets)}
        document_rows = []
        for target in targets:
            label = target["label"]
            for doc_id, cnt in profiles[label]["doc_counts"].items():
                year, title = meta_for(doc_id)
                document_rows.append({
                    "term": label,
                    "doc_id": doc_id,
                    "year": year,
                    "title": title,
                    "n_windows": cnt,
                    "in_seed": "yes" if doc_id in seed_docs else "no",
                })
        document_rows.sort(
            key=lambda r: (term_order[r["term"]], r["year"] or 0, str(r["doc_id"]))
        )

        for r in decade_document_rows:
            r["year"], r["title"] = meta_for(r["doc_id"])
        decade_document_rows.sort(
            key=lambda r: (r["decade"], term_order[r["term"]],
                           r["year"] or 0, str(r["doc_id"]))
        )

        write_csv(
            Path(str(prefix) + "_documents.csv"),
            document_rows,
            ["term", "doc_id", "year", "title", "n_windows", "in_seed"],
        )
        if args.match_decades:
            write_csv(
                Path(str(prefix) + "_decade_documents.csv"),
                decade_document_rows,
                ["decade", "term", "doc_id", "year", "title",
                 "n_windows", "in_seed"],
            )

        # Text windows (the passage around each hit)
        window_rows = []
        for target in targets:
            label = target["label"]
            window_rows.extend(build_window_rows(
                label, profiles[label]["windows"], profiles[label]["hit_meta"],
                collocates, excluded_keys, limit=args.max_window_rows,
            ))
        for r in window_rows:
            r["title"] = meta_for(r["doc_id"])[1]
        window_rows.sort(
            key=lambda r: (term_order[r["term"]], r["year"] or 0,
                           str(r["doc_id"]), r["token_idx"])
        )
        window_fields = ["term", "doc_id", "year", "title", "token_idx",
                         "surface", "collocates_present", "passage"]
        write_csv(Path(str(prefix) + "_windows.csv"), window_rows, window_fields)

        if args.match_decades:
            for r in decade_window_rows:
                r["title"] = meta_for(r["doc_id"])[1]
            decade_window_rows.sort(
                key=lambda r: (r["decade"], term_order[r["term"]],
                               r["year"] or 0, str(r["doc_id"]), r["token_idx"])
            )
            write_csv(
                Path(str(prefix) + "_decade_windows.csv"),
                decade_window_rows,
                ["decade"] + window_fields,
            )

        # Collocate hits: one row per (collocate, term, window), so that e.g.
        # every 'resist' + 'violence' window sits next to every 'withstand' +
        # 'violence' window. Always built from the full window set, ignoring
        # --max-window-rows, so pair lists are never silently truncated.
        if args.max_window_rows:
            hit_source = []
            for target in targets:
                label = target["label"]
                hit_source.extend(build_window_rows(
                    label, profiles[label]["windows"],
                    profiles[label]["hit_meta"], collocates, excluded_keys,
                    limit=0,
                ))
            for r in hit_source:
                r["title"] = meta_for(r["doc_id"])[1]
        else:
            hit_source = window_rows

        collocate_order = {c["word"]: i for i, c in enumerate(collocates)}
        collocate_hit_rows = []
        for r in hit_source:
            for word in filter(None, r["collocates_present"].split(";")):
                collocate_hit_rows.append({
                    "collocate": word,
                    "term": r["term"],
                    "doc_id": r["doc_id"],
                    "year": r["year"],
                    "title": r["title"],
                    "token_idx": r["token_idx"],
                    "surface": r["surface"],
                    "passage": r["passage"],
                })
        collocate_hit_rows.sort(
            key=lambda r: (collocate_order[r["collocate"]],
                           term_order[r["term"]], r["year"] or 0,
                           str(r["doc_id"]), r["token_idx"])
        )
        write_csv(
            Path(str(prefix) + "_collocate_hits.csv"),
            collocate_hit_rows,
            ["collocate", "term", "doc_id", "year", "title", "token_idx",
             "surface", "passage"],
        )

        def log_windows(label):
            rows = [r for r in window_rows if r["term"] == label]
            if not rows or args.log_windows < 1:
                return
            step = max(1, len(rows) // args.log_windows)
            sample = rows[::step][:args.log_windows]
            logger.info(f"\nExample windows for {label} "
                        f"({len(sample)} of {len(rows)}):")
            for r in sample:
                logger.info(f"  [{r['year']}] {r['title'][:50]} "
                            f"(doc {r['doc_id']}, token {r['token_idx']})")
                if r["collocates_present"]:
                    logger.info(f"    collocates: {r['collocates_present']}")
                logger.info(f"    {r['passage']}")

        def log_documents(label):
            docs = sorted(
                profiles[label]["doc_counts"].items(),
                key=lambda kv: (meta_for(kv[0])[0] or 0, str(kv[0])),
            )
            shown = docs if args.log_documents == 0 else docs[:args.log_documents]
            logger.info(f"\nDocuments for {label} ({len(docs)}):")
            for doc_id, cnt in shown:
                year, title = meta_for(doc_id)
                flag = "" if label == seed or doc_id in seed_docs else "  [not in seed]"
                logger.info(f"  {year!s:<5} {doc_id!s:<12} {title[:70]:<70} "
                            f"({cnt} win){flag}")
            if len(docs) > len(shown):
                logger.info(f"  ... {len(docs) - len(shown)} more (see _documents.csv)")

        for target in targets:
            log_documents(target["label"])
            log_windows(target["label"])

        # ------------------------------------------------------------------
        # 5b. Remaining CSV outputs
        # ------------------------------------------------------------------
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
             "mlm_rank", "mlm_score", "match_documents", "match_decades"],
        )
        write_csv(
            Path(str(prefix) + "_collocate_surfaces.csv"),
            surface_rows,
            ["term", "collocate_label", "collocate_key",
             "observed_surface", "count"],
        )

        if args.match_decades:
            write_csv(
                Path(str(prefix) + "_decade_summary.csv"),
                decade_summary_rows,
                ["decade", "term", "forms", "n_windows", "n_docs",
                 "cosine_to_seed_profile", "mlm_rank", "mlm_score"],
            )
            write_csv(
                Path(str(prefix) + "_decade_matrix.csv"),
                decade_matrix_rows,
                ["decade", "term", "forms", "n_windows", "collocate",
                 "cooccurrence_windows", "window_rate"],
            )

        # Console summary
        logger.info(f"\nSeed profile windows: {seed_n} across {len(seed_docs)} documents")
        if args.match_documents:
            logger.info("(Global substitutes restricted to these documents)")
        logger.info("Global profile similarity to seed (selected-collocate cosine):")
        for target in sorted(
            substitutes,
            key=lambda t: (
                cosine(seed_rates, profiles[t["label"]]["rates"])
                if profiles[t["label"]]["n"] >= args.min_windows else -1
            ),
            reverse=True,
        ):
            prof = profiles[target["label"]]
            score = (
                f"{cosine(seed_rates, prof['rates']):.3f}"
                if prof["n"] >= args.min_windows else "n/a"
            )
            logger.info(
                f"  {target['label']:<26} {score:>6} "
                f"({prof['n']} windows, {len(prof['doc_ids'])} docs)"
            )

        if args.match_decades and decade_summary_rows:
            logger.info("\nPer-decade cosines (only decades with enough seed windows):")
            rows_by_dec = defaultdict(list)
            for r in decade_summary_rows:
                if r["term"] != seed:
                    rows_by_dec[r["decade"]].append(r)
            for dec in sorted(rows_by_dec):
                logger.info(f"  {dec}:")
                for r in sorted(
                    rows_by_dec[dec],
                    key=lambda x: float(x["cosine_to_seed_profile"] or -1),
                    reverse=True,
                ):
                    cos = r["cosine_to_seed_profile"] or "n/a"
                    logger.info(f"    {r['term']:<24} {cos:>8} "
                          f"({r['n_windows']} win, {r['n_docs']} docs)")

        logger.info(f"\nSaved: {prefix}_matrix.csv")
        logger.info(f"Saved: {prefix}_detail.csv")
        logger.info(f"Saved: {prefix}_summary.csv")
        logger.info(f"Saved: {prefix}_collocate_surfaces.csv")
        logger.info(f"Saved: {prefix}_documents.csv")
        logger.info(f"Saved: {prefix}_windows.csv")
        logger.info(f"Saved: {prefix}_collocate_hits.csv")
        if args.match_decades:
            logger.info(f"Saved: {prefix}_decade_windows.csv")
            logger.info(f"Saved: {prefix}_decade_summary.csv")
            logger.info(f"Saved: {prefix}_decade_matrix.csv")
            logger.info(f"Saved: {prefix}_decade_documents.csv")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
