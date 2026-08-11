#!/usr/bin/env python
"""
observation_store_parity.py — Compare two Tier 1 observation stores

Verifies that a Zarr store and a Parquet store (or any two backends)
hold equivalent observations. Designed for dual-write / migration checks
before flipping FAISS or Tier 2 to the new backend.

Checks (in order)
-----------------
1. Cardinality: n_events, year_bounds, doc_key set size
2. Identity:     event_id set equality (with optional size cap reporting)
3. Metadata:     for a sample of shared event_ids, field-wise equality
4. Embeddings:   for a sample of shared event_ids, per-scale cosine / L2
                 agreement within a configurable tolerance

Usage
-----
    python observation_store_parity.py \\
        --left zarr:/data/tier1 \\
        --right parquet:/data/tier1_parquet

    python observation_store_parity.py \\
        --left zarr:/data/tier1 \\
        --right parquet:/data/tier1_parquet \\
        --sample 500 --atol 1e-5 --rtol 1e-4

Exit codes
----------
    0  all checks passed
    1  one or more checks failed
    2  usage / setup error
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from observation_store_api import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    open_observation_lookup,
    open_observation_stream,
    open_observation_writer,
)

# Register backends that are available in this environment.
for _mod in ("zarr_observation_backend", "parquet_observation_backend"):
    try:
        __import__(_mod)
    except ImportError as _exc:
        print(f"[parity] optional backend not loaded: {_mod} ({_exc})")


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

def parse_store_spec(spec: str) -> tuple[str, Path]:
    """
    Parse 'backend:path' or bare path (defaults to zarr).

    Examples:
        zarr:/data/tier1
        parquet:/data/tier1_parquet
        /data/tier1          → ('zarr', Path('/data/tier1'))
    """
    if ":" in spec and not (len(spec) > 1 and spec[1] == ":"):
        # Windows drive letters look like 'C:\...' — only split on first
        # colon when the left side is a known backend name.
        backend, _, rest = spec.partition(":")
        if backend in ("zarr", "parquet") and rest:
            return backend, Path(rest)
    return "zarr", Path(spec)


# ---------------------------------------------------------------------------
# Result accumulation
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        extra = f" — {self.detail}" if self.detail else ""
        return f"[{status}] {self.name}{extra}"


@dataclass
class ParityReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        print(result.line())
        if result.stats:
            for k, v in result.stats.items():
                print(f"       {k}: {v}")

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_id_map_from_stream(
    stream,
    *,
    max_events: Optional[int] = None,
    year_filter: Optional[set[int]] = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Stream multi-scale embeddings into
        event_id → (emb_local, emb_medium, emb_broad, pub_year)

    If max_events is set, stop after that many unique ids (useful for
    quick smoke tests on huge corpora).
    """
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for emb_l, emb_m, emb_b, eids, years in stream.iter_multi_scale_embeddings(
        batch_size=8192,
        year_filter=year_filter,
    ):
        for i in range(len(eids)):
            eid = int(eids[i])
            if eid in out:
                continue
            out[eid] = (
                np.asarray(emb_l[i], dtype=np.float32),
                np.asarray(emb_m[i], dtype=np.float32),
                np.asarray(emb_b[i], dtype=np.float32),
                int(years[i]),
            )
            if max_events is not None and len(out) >= max_events:
                return out
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ensemble(local, medium, broad, weights=DEFAULT_ENSEMBLE_WEIGHTS) -> np.ndarray:
    return (
        weights[0] * local + weights[1] * medium + weights[2] * broad
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_cardinality(left_spec, right_spec, report: ParityReport) -> tuple[Any, Any]:
    lb, lp = left_spec
    rb, rp = right_spec

    # Writers expose n_events / get_doc_keys without loading embeddings.
    # dim is required by the factory but unused for read-only queries on
    # an existing store; pass a placeholder.
    left_w = open_observation_writer(lb, lp, dim=768)
    right_w = open_observation_writer(rb, rp, dim=768)

    ln, rn = left_w.n_events, right_w.n_events
    report.add(
        CheckResult(
            "n_events",
            ln == rn,
            f"left={ln:,} right={rn:,}",
            {"delta": abs(ln - rn)},
        )
    )

    try:
        left_s = open_observation_stream(lb, lp)
        right_s = open_observation_stream(rb, rp)
        lbounds = left_s.year_bounds()
        rbounds = right_s.year_bounds()
        report.add(
            CheckResult(
                "year_bounds",
                lbounds == rbounds,
                f"left={lbounds} right={rbounds}",
            )
        )
    except Exception as exc:
        report.add(CheckResult("year_bounds", False, f"error: {exc}"))
        left_s = right_s = None

    try:
        ldocs = left_w.get_doc_keys()
        rdocs = right_w.get_doc_keys()
        report.add(
            CheckResult(
                "doc_keys",
                ldocs == rdocs,
                f"left={len(ldocs):,} right={len(rdocs):,} "
                f"only_left={len(ldocs - rdocs):,} only_right={len(rdocs - ldocs):,}",
            )
        )
    except Exception as exc:
        report.add(CheckResult("doc_keys", False, f"error: {exc}"))

    return left_s, right_s


def check_event_ids(
    left_spec,
    right_spec,
    report: ParityReport,
    *,
    max_ids: Optional[int] = None,
) -> tuple[set[int], set[int]]:
    lb, lp = left_spec
    rb, rp = right_spec
    left_w = open_observation_writer(lb, lp, dim=768)
    right_w = open_observation_writer(rb, rp, dim=768)

    left_ids = left_w.get_event_ids()
    right_ids = right_w.get_event_ids()

    only_left = left_ids - right_ids
    only_right = right_ids - left_ids
    shared = left_ids & right_ids

    ok = len(only_left) == 0 and len(only_right) == 0
    detail = (
        f"shared={len(shared):,} only_left={len(only_left):,} "
        f"only_right={len(only_right):,}"
    )
    stats: dict[str, Any] = {}
    if only_left:
        sample = sorted(only_left)[:10]
        stats["only_left_sample"] = sample
    if only_right:
        sample = sorted(only_right)[:10]
        stats["only_right_sample"] = sample

    report.add(CheckResult("event_id_set", ok, detail, stats))
    return left_ids, right_ids


def check_metadata_sample(
    left_spec,
    right_spec,
    shared_ids: set[int],
    report: ParityReport,
    *,
    sample: int = 200,
    seed: int = 0,
) -> list[int]:
    if not shared_ids:
        report.add(CheckResult("metadata_sample", False, "no shared event_ids"))
        return []

    rng = np.random.default_rng(seed)
    sample_ids = list(shared_ids)
    if len(sample_ids) > sample:
        sample_ids = list(rng.choice(sample_ids, size=sample, replace=False))
    sample_ids = [int(x) for x in sample_ids]

    lb, lp = left_spec
    rb, rp = right_spec
    left_lookup = open_observation_lookup(lb, lp)
    right_lookup = open_observation_lookup(rb, rp)

    meta_fields = (
        "vector_id",
        "corpus",
        "doc_id",
        "token",
        "token_idx",
        "window_id",
        "window_token_pos",
        "pub_year",
    )

    mismatches = 0
    missing_left = 0
    missing_right = 0
    examples: list[str] = []

    for eid in sample_ids:
        try:
            lm = left_lookup.get_event_metadata(eid)
        except KeyError:
            missing_left += 1
            continue
        try:
            rm = right_lookup.get_event_metadata(eid)
        except KeyError:
            missing_right += 1
            continue

        for f in meta_fields:
            lv = lm.get(f)
            rv = rm.get(f)
            # Normalise string comparisons
            if isinstance(lv, str) or isinstance(rv, str):
                lv, rv = str(lv) if lv is not None else lv, str(rv) if rv is not None else rv
            if lv != rv:
                mismatches += 1
                if len(examples) < 8:
                    examples.append(f"event_id={eid} field={f} left={lv!r} right={rv!r}")
                break  # one mismatch per event is enough for counting

    ok = mismatches == 0 and missing_left == 0 and missing_right == 0
    report.add(
        CheckResult(
            "metadata_sample",
            ok,
            f"sampled={len(sample_ids)} mismatches={mismatches} "
            f"missing_left={missing_left} missing_right={missing_right}",
            {"examples": examples} if examples else {},
        )
    )
    return sample_ids


def check_embeddings_sample(
    left_stream,
    right_stream,
    sample_ids: list[int],
    report: ParityReport,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    min_cosine: float = 0.9999,
):
    if not sample_ids:
        report.add(CheckResult("embeddings_sample", False, "no sample ids"))
        return

    want = set(sample_ids)
    # Stream until we have collected all sample ids from both sides
    # (or the stream is exhausted).
    left_map: dict[int, tuple] = {}
    right_map: dict[int, tuple] = {}

    for emb_l, emb_m, emb_b, eids, years in left_stream.iter_multi_scale_embeddings(
        batch_size=8192
    ):
        for i in range(len(eids)):
            eid = int(eids[i])
            if eid in want and eid not in left_map:
                left_map[eid] = (emb_l[i], emb_m[i], emb_b[i], int(years[i]))
        if len(left_map) >= len(want):
            break

    for emb_l, emb_m, emb_b, eids, years in right_stream.iter_multi_scale_embeddings(
        batch_size=8192
    ):
        for i in range(len(eids)):
            eid = int(eids[i])
            if eid in want and eid not in right_map:
                right_map[eid] = (emb_l[i], emb_m[i], emb_b[i], int(years[i]))
        if len(right_map) >= len(want):
            break

    missing_l = want - set(left_map)
    missing_r = want - set(right_map)

    scale_fail = {"local": 0, "medium": 0, "broad": 0, "ensemble": 0}
    year_fail = 0
    worst: list[str] = []
    compared = 0

    for eid in sample_ids:
        if eid not in left_map or eid not in right_map:
            continue
        ll, lm, lb_, ly = left_map[eid]
        rl, rm, rb_, ry = right_map[eid]
        compared += 1

        if ly != ry:
            year_fail += 1

        for name, a, b in (
            ("local", ll, rl),
            ("medium", lm, rm),
            ("broad", lb_, rb_),
        ):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            cos = _cosine(a, b)
            close = np.allclose(a, b, atol=atol, rtol=rtol)
            if not close or cos < min_cosine:
                scale_fail[name] += 1
                if len(worst) < 10:
                    max_abs = float(np.max(np.abs(a - b)))
                    worst.append(
                        f"event_id={eid} scale={name} cos={cos:.8f} max_abs={max_abs:.3e}"
                    )

        le = _ensemble(ll, lm, lb_)
        re = _ensemble(rl, rm, rb_)
        cos_e = _cosine(le, re)
        if not np.allclose(le, re, atol=atol, rtol=rtol) or cos_e < min_cosine:
            scale_fail["ensemble"] += 1
            if len(worst) < 10:
                worst.append(
                    f"event_id={eid} scale=ensemble cos={cos_e:.8f}"
                )

    ok = (
        compared > 0
        and all(v == 0 for v in scale_fail.values())
        and year_fail == 0
        and not missing_l
        and not missing_r
    )
    report.add(
        CheckResult(
            "embeddings_sample",
            ok,
            f"compared={compared} year_mismatch={year_fail} "
            f"missing_left={len(missing_l)} missing_right={len(missing_r)} "
            f"scale_fails={scale_fail}",
            {
                "atol": atol,
                "rtol": rtol,
                "min_cosine": min_cosine,
                "worst": worst,
            }
            if not ok
            else {"compared": compared},
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(
        description="Parity-check two Tier 1 observation stores (e.g. Zarr vs Parquet)."
    )
    p.add_argument(
        "--left",
        required=True,
        help="Left store as backend:path (e.g. zarr:/data/tier1)",
    )
    p.add_argument(
        "--right",
        required=True,
        help="Right store as backend:path (e.g. parquet:/data/tier1_parquet)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Number of shared event_ids to sample for metadata/embedding checks (default 200)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    p.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for embedding allclose (default 1e-5)",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for embedding allclose (default 1e-4)",
    )
    p.add_argument(
        "--min-cosine",
        type=float,
        default=0.9999,
        help="Minimum cosine similarity per scale (default 0.9999)",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embedding sample check (metadata + identity only)",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip the metadata sample check",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        left_spec = parse_store_spec(args.left)
        right_spec = parse_store_spec(args.right)
    except Exception as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 2

    print(f"left : {left_spec[0]}:{left_spec[1]}")
    print(f"right: {right_spec[0]}:{right_spec[1]}")
    print()

    report = ParityReport()

    left_stream, right_stream = check_cardinality(left_spec, right_spec, report)
    left_ids, right_ids = check_event_ids(left_spec, right_spec, report)
    shared = left_ids & right_ids

    sample_ids: list[int] = []
    if not args.skip_metadata:
        sample_ids = check_metadata_sample(
            left_spec,
            right_spec,
            shared,
            report,
            sample=args.sample,
            seed=args.seed,
        )
    else:
        # Still need sample ids for embeddings if requested
        if shared and not args.skip_embeddings:
            rng = np.random.default_rng(args.seed)
            pool = list(shared)
            n = min(args.sample, len(pool))
            sample_ids = [int(x) for x in rng.choice(pool, size=n, replace=False)]

    if not args.skip_embeddings:
        if left_stream is None:
            left_stream = open_observation_stream(left_spec[0], left_spec[1])
        if right_stream is None:
            right_stream = open_observation_stream(right_spec[0], right_spec[1])
        check_embeddings_sample(
            left_stream,
            right_stream,
            sample_ids,
            report,
            atol=args.atol,
            rtol=args.rtol,
            min_cosine=args.min_cosine,
        )

    print()
    if report.ok:
        print("RESULT: PASS — stores are parity-equivalent under the configured checks")
        return 0
    print("RESULT: FAIL — see checks above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
