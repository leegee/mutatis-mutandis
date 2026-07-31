#!/usr/bin/env python
"""
tier1_2_masked_concept_embeddings.py - masked-target contextual embeddings,
scoped to concept-matched token occurrences only.

Background
----------
Tier 1 (tier1_corpus2zarr.py) stores hidden[i] - the raw contextualized
hidden state at each content token's OWN position - for every content
token in the corpus, at three window scales. That vector still carries a
strong residual signal from the token's own input embedding even after
many transformer layers (a known property of BERT-family hidden states),
which is the direct explanation for concept clusters separating by
spelling ("liberty" vs "libertyes") rather than by sense - the clustering
was always partly clustering on lexical identity baked into the vector,
not purely on context.

The fix - replacing the target token with [MASK] (all its WordPiece
subword pieces) before extracting a vector - can't reuse Tier 1's
exhaustive per-token pass efficiently: Tier 1's speed comes from one
forward pass yielding hidden states for every token in a window
simultaneously, whereas masked extraction needs a distinct masked input
per target token (mask token T, leave everything else visible). Doing
this for every content token in the corpus would multiply Tier 1's
compute cost by roughly the average window length.

This script instead scopes masking to ONLY the token occurrences that
already matched a tracked concept's forms - the same set Tier 3's
clustering operates on (lookup.iter_matching_event_ids(...)) - since
those are the only vectors this fix is actually meant to improve, and
non-concept tokens' masked embeddings would never be consumed anyway.

Grouped by doc, occurrences are batched into padded forward passes per
window-size config for throughput, rather than one forward pass per
occurrence.

Output
------
A separate Zarr store (MASKED_ZARR_PATH, distinct from Tier 1's
ZARR_PATH) holding, per event_id already known from Tier 2/3:
    - emb_masked_local, emb_masked_medium, emb_masked_broad

Kept as an additive, separate artifact rather than overwriting Tier 1's
existing emb_local/medium/broad, so this can be validated (does
clustering on masked vectors actually separate by sense rather than
spelling?) before deciding whether it replaces or supplements the
existing embeddings in EmbeddingCache.

Usage:
    python tier1_2_masked_concept_embeddings.py --concept liberty
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch

from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.eebo_config import MASKED_ZARR_PATH, EMBED_BATCH_SIZE
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.macberth import load_macberth
from lib.corpus_db import sqlite3_connection  # matches usage pattern in tier2/tier3
from lib.eebo_config import CORPUS_TIER2_DB_PATH

from tier1_0_corpus2zarr import (
    WINDOW_CONFIGS,
    is_content_token,
    DocBuffer,
    stable_hash,
)


@dataclass
class RunStats:
    """
    Accumulates counts across the whole run so a single structured
    summary can be logged at the end, instead of only ever seeing
    scattered per-event warnings with no final tally of what was
    actually written vs skipped.
    """
    concept: str
    total_targets: int = 0
    docs_total: int = 0
    docs_skipped_already_done: int = 0
    docs_skipped_empty_buffer: int = 0
    events_no_buffer_pos: int = 0          # token_idx filtered out upstream (stopword/punct)
    events_no_window: dict[str, int] = None  # per-config: no window found
    events_no_mask_positions: dict[str, int] = None  # per-config: word not found in window's word_ids
    events_missing_config: int = 0          # had some configs but not all three
    events_written: int = 0

    def __post_init__(self):
        self.events_no_window = defaultdict(int)
        self.events_no_mask_positions = defaultdict(int)

    def log_summary(self):
        logger.info(
            f"[tier1b] === run summary: concept={self.concept} ===\n"
            f"    target occurrences fetched : {self.total_targets}\n"
            f"    docs total                 : {self.docs_total}\n"
            f"    docs skipped (already done): {self.docs_skipped_already_done}\n"
            f"    docs skipped (empty buffer): {self.docs_skipped_empty_buffer}\n"
            f"    events skipped (no buffer pos, filtered upstream): {self.events_no_buffer_pos}\n"
            f"    events skipped (no window found)      : {dict(self.events_no_window)}\n"
            f"    events skipped (no mask positions)     : {dict(self.events_no_mask_positions)}\n"
            f"    events skipped (missing 1+ config)     : {self.events_missing_config}\n"
            f"    events successfully written            : {self.events_written}"
        )
        if self.total_targets and self.events_written < self.total_targets * 0.5:
            logger.warning(
                f"[tier1b] concept={self.concept}: only "
                f"{self.events_written}/{self.total_targets} "
                f"({self.events_written / self.total_targets:.0%}) target occurrences "
                f"produced a written vector - worth investigating before trusting "
                f"clustering results built from this store."
            )

@dataclass
class TargetOccurrence:
    event_id: int
    doc_id: str
    token_idx: int   # corpus token_idx, matches DocBuffer.corpus_token_idxs
    token: str
    vector_id: int
    # window_id/window_token_pos as originally recorded by Tier 1 - this is
    # the MEDIUM-scale window (Tier 1's _flush uses "medium" as canonical
    # when present), preserved so the medium-config masked vector can reuse
    # the exact same window rather than a re-derived approximation.
    orig_window_id: int
    orig_window_token_pos: int


def fetch_concept_targets(sqlite_db_path, concept: str) -> list[TargetOccurrence]:
    """
    Pull the exact set of occurrences already known to match this concept,
    from Tier 2/3's sqlite events table - reusing the concept resolution
    that already happened, rather than re-matching forms against Postgres
    here.
    """
    con = sqlite3_connection(sqlite_db_path)
    rows = con.execute(
        """
        SELECT event_id, doc_id, token_idx, token, vector_id, window_id, window_token_pos
        FROM events
        WHERE concept = ?
        ORDER BY doc_id, token_idx
        """,
        (concept,),
    ).fetchall()
    con.close()

    return [
        TargetOccurrence(
            event_id=r[0], doc_id=r[1], token_idx=r[2], token=r[3],
            vector_id=r[4], orig_window_id=r[5], orig_window_token_pos=r[6],
        )
        for r in rows
    ]


# doc buffer reconstruction (mirrors tier1_corpus2zarr.py exactly)

def build_doc_buffer(pg_conn, doc_id: str) -> DocBuffer:
    """
    Rebuild the SAME content-token buffer Tier 1 built for this doc, so
    buffer positions/word_ids line up identically. Must use the identical
    is_content_token filter as Tier 1, or word_ids alignment will silently
    drift out of sync with the original embeddings' indexing.
    """
    buf = DocBuffer(doc_id=doc_id)
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT token_idx, vector_id, token
            FROM pamphlet_tokens
            WHERE doc_id = %s
            ORDER BY token_idx
            """,
            (doc_id,),
        )
        for token_idx, vector_id, token in cur:
            if is_content_token(token):
                buf.append(token, vector_id, token_idx)
    return buf


# --- window selection + masking ---------------------------------------------

@dataclass
class MaskedWindowJob:
    event_id: int
    config_name: str
    input_ids: list[int]
    attention_mask: list[int]
    mask_positions: list[int]   # positions WITHIN this window's input_ids


def exact_window_from_stored_id(
    word_ids: list[int | None],
    window_start_word: int,
    window_size: int,
) -> tuple[int, int] | None:
    """
    Reconstruct the SAME window Tier 1 used, given the stored window_id
    (= start_word in word-space, per tier1_corpus2zarr.py's _flush, which
    sets window_ids.append(canonical.window_start)). Only meaningful for
    the "medium" config, since that's the only scale whose window survived
    into the events table (canonical = medium when present). Using this
    instead of find_window_containing means the medium masked vector
    corresponds to the EXACT same observation as the original, not a
    plausible substitute.
    """
    n = len(word_ids)
    try:
        encoded_start = next(i for i, wid in enumerate(word_ids) if wid == window_start_word)
    except StopIteration:
        return None
    encoded_end = min(encoded_start + window_size, n)
    return (encoded_start, encoded_end)


def find_window_containing(
    word_ids: list[int | None],
    target_buffer_idx: int,
    window_size: int,
    stride: int,
) -> tuple[int, int] | None:
    """
    Find an (encoded_start, encoded_end) span, sized to window_size, whose
    word range includes target_buffer_idx - preferring the window that
    centers the target most closely, since a target sitting near a
    window's edge gets less surrounding context on one side than the
    other. Mirrors Tier 1's _iter_windows_config stepping logic but stops
    at the best-centered candidate instead of yielding every window.
    """
    n = len(word_ids)
    valid = [wid for wid in word_ids if wid is not None and wid >= 0]
    if not valid:
        return None
    n_words = max(valid) + 1
    if target_buffer_idx >= n_words:
        return None

    best = None
    best_centering = None

    start_word = 0
    while start_word < n_words:
        try:
            encoded_start = next(i for i, wid in enumerate(word_ids) if wid == start_word)
        except StopIteration:
            break
        encoded_end = min(encoded_start + window_size, n)

        window_word_ids = word_ids[encoded_start:encoded_end]
        window_word_set = {wid for wid in window_word_ids if wid is not None and wid >= 0}

        if target_buffer_idx in window_word_set:
            span_words = [wid for wid in window_word_ids if wid is not None and wid >= 0]
            mid = (span_words[0] + span_words[-1]) / 2
            centering = abs(target_buffer_idx - mid)
            if best_centering is None or centering < best_centering:
                best = (encoded_start, encoded_end)
                best_centering = centering

        if encoded_end == n:
            break
        start_word += stride

    return best


def build_masked_job(
    tokenizer,
    input_ids: list[int],
    word_ids: list[int | None],
    span: tuple[int, int],
    target_buffer_idx: int,
    event_id: int,
    config_name: str,
) -> MaskedWindowJob | None:
    encoded_start, encoded_end = span
    window_ids = input_ids[encoded_start:encoded_end]
    window_word_ids = word_ids[encoded_start:encoded_end]

    # All subtoken positions (within this window) belonging to the target
    # word - whole-word masking. Reusing word_ids here (already computed
    # by Tier 1's tokenizer call) is simpler and more reliable than
    # re-deriving subword spans via character offsets.
    mask_positions = [
        i for i, wid in enumerate(window_word_ids) if wid == target_buffer_idx
    ]
    if not mask_positions:
        return None

    masked_ids = list(window_ids)
    for pos in mask_positions:
        masked_ids[pos] = tokenizer.mask_token_id

    return MaskedWindowJob(
        event_id=event_id,
        config_name=config_name,
        input_ids=masked_ids,
        attention_mask=[1] * len(masked_ids),
        mask_positions=mask_positions,
    )


# --- batched forward pass ----------------------------------------------------

def run_batch(model, device, jobs: list[MaskedWindowJob], pooling_scope: str = "mask_only") -> dict[int, np.ndarray]:
    """
    One padded forward pass for a batch of same-config jobs. Returns
    {event_id: pooled_vector}.

    pooling_scope:
      "mask_only" - pool only the masked position(s). Tightest "what fills
                    this slot" signal; closest replacement for Tier 1's
                    existing hidden[i] extraction at the target's own
                    position, just computed under masking instead.
      "context"   - pool all non-masked, non-padding positions instead.
                    Zero contribution from the target position at all.
    """
    if not jobs:
        return {}

    max_len = max(len(j.input_ids) for j in jobs)

    def pad(seq, pad_value=0):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = torch.tensor([pad(j.input_ids) for j in jobs], dtype=torch.long).to(device)
    attention_mask = torch.tensor([pad(j.attention_mask) for j in jobs], dtype=torch.long).to(device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    hidden = out.last_hidden_state.cpu().numpy()  # (batch, seq_len, hidden_dim)

    results = {}
    for b, job in enumerate(jobs):
        if pooling_scope == "context":
            valid_len = sum(job.attention_mask)
            pool_idxs = [i for i in range(valid_len) if i not in job.mask_positions]
        else:  # mask_only
            pool_idxs = job.mask_positions

        if not pool_idxs:
            logger.warning(f"[tier1b] event {job.event_id}: no positions to pool, skipping")
            continue

        results[job.event_id] = hidden[b, pool_idxs].mean(axis=0).astype(np.float32)

    return results


# --- orchestration ------------------------------------------------------------

def process_concept(concept: str, pooling_scope: str = "mask_only", batch_size: int = EMBED_BATCH_SIZE):
    """
    NOTE on field naming: this writes into a SEPARATE zarr store
    (MASKED_ZARR_PATH, not Tier 1's ZARR_PATH), but reuses the same
    ZarrEmbeddingObservationStore class - whose emb_local/emb_medium/
    emb_broad field names are hardcoded regardless of what's actually
    written into them. In THIS store those three fields hold MASKED
    vectors, not raw hidden states. That's fine as long as callers know
    which path they're reading from, but worth flagging clearly since the
    field names alone don't distinguish "raw" vs "masked" - only the
    store's path does.
    """
    logger.info(f"[tier1b] loading targets for concept={concept}")
    targets = fetch_concept_targets(CORPUS_TIER2_DB_PATH, concept)
    if not targets:
        logger.warning(f"[tier1b] no events found for concept={concept}")
        return

    stats = RunStats(concept=concept, total_targets=len(targets))

    logger.info(f"[tier1b] {len(targets)} target occurrences, grouping by doc")
    by_doc: dict[str, list[TargetOccurrence]] = defaultdict(list)
    for t in targets:
        by_doc[t.doc_id].append(t)
    stats.docs_total = len(by_doc)

    mac = load_macberth()
    pg_conn = get_connection()

    store = ZarrEmbeddingObservationStore(
        path=str(MASKED_ZARR_PATH),
        dim=mac.model.config.hidden_size,
    )
    already_done = store.get_doc_ids()

    docs_processed = 0
    for doc_id, occurrences in by_doc.items():
        if doc_id in already_done:
            stats.docs_skipped_already_done += 1
            continue

        buf = build_doc_buffer(pg_conn, doc_id)
        if not buf:
            logger.warning(f"[tier1b] doc {doc_id}: empty buffer, skipping")
            stats.docs_skipped_empty_buffer += 1
            continue

        idx_to_buffer_pos = {ci: pos for pos, ci in enumerate(buf.corpus_token_idxs)}

        enc = mac.tokenizer(buf.tokens, is_split_into_words=True, truncation=False, return_tensors="pt")
        input_ids = enc["input_ids"][0].tolist()
        word_ids = enc.word_ids() or [None] * len(input_ids)

        # Build all jobs for this doc across all configs, then batch by
        # config for the forward passes, then assemble complete per-event
        # rows (all three configs present) before a single append_events
        # call - the store requires local+medium+broad together, so
        # partial per-config writes aren't an option here.
        jobs_by_config: dict[str, list[MaskedWindowJob]] = defaultdict(list)
        valid_occurrences: list[TargetOccurrence] = []

        for occ in occurrences:
            buffer_pos = idx_to_buffer_pos.get(occ.token_idx)
            if buffer_pos is None:
                logger.warning(
                    f"[tier1b] event {occ.event_id}: token_idx {occ.token_idx} "
                    f"not in content-token buffer for doc {doc_id} (filtered as stopword/punct?), skipping"
                )
                stats.events_no_buffer_pos += 1
                continue

            occ_ok = True
            for config in WINDOW_CONFIGS:
                if config["name"] == "medium":
                    # Exact reconstruction of the original observation's
                    # window, using window_id as stored in sqlite.
                    span = exact_window_from_stored_id(word_ids, occ.orig_window_id, config["size"])
                else:
                    # local/broad windows were never preserved in sqlite -
                    # this is an approximation (best-centered window),
                    # not the exact original observation.
                    span = find_window_containing(word_ids, buffer_pos, config["size"], config["stride"])

                if span is None:
                    logger.warning(
                        f"[tier1b] event {occ.event_id}: no window found for config={config['name']}"
                    )
                    stats.events_no_window[config["name"]] += 1
                    occ_ok = False
                    continue

                job = build_masked_job(
                    mac.tokenizer, input_ids, word_ids, span, buffer_pos,
                    occ.event_id, config["name"],
                )
                if job is None:
                    stats.events_no_mask_positions[config["name"]] += 1
                    occ_ok = False
                    continue

                jobs_by_config[config["name"]].append(job)

            if occ_ok:
                valid_occurrences.append(occ)

        # Run batched forward passes per config, collecting event_id -> vector.
        vectors_by_config: dict[str, dict[int, np.ndarray]] = {}
        for config in WINDOW_CONFIGS:
            jobs = jobs_by_config[config["name"]]
            collected: dict[int, np.ndarray] = {}
            for i in range(0, len(jobs), batch_size):
                chunk = jobs[i : i + batch_size]
                collected.update(run_batch(mac.model, mac.device, chunk, pooling_scope=pooling_scope))
            vectors_by_config[config["name"]] = collected

        write_doc_events(store, valid_occurrences, vectors_by_config, stats)

        docs_processed += 1
        if docs_processed % 50 == 0:
            logger.info(f"[tier1b] processed {docs_processed}/{len(by_doc)} docs")

    pg_conn.close()
    logger.info(f"[tier1b] done: concept={concept}, docs={docs_processed}")
    stats.log_summary()


def write_doc_events(
    store: ZarrEmbeddingObservationStore,
    occurrences: list[TargetOccurrence],
    vectors_by_config: dict[str, dict[int, np.ndarray]],
    stats: "RunStats",
):
    """
    Assemble and write one append_events call covering every occurrence
    in this doc that has a vector for all three configs. Occurrences
    missing any one config (e.g. no window found) are skipped and logged,
    rather than written with a zero-filled placeholder - a silently
    zeroed vector would be indistinguishable from a real embedding
    downstream and could corrupt clustering without any visible signal.
    """
    dim = None
    rows = []
    for occ in occurrences:
        local = vectors_by_config["local"].get(occ.event_id)
        medium = vectors_by_config["medium"].get(occ.event_id)
        broad = vectors_by_config["broad"].get(occ.event_id)

        if local is None or medium is None or broad is None:
            missing = [
                name for name, v in
                [("local", local), ("medium", medium), ("broad", broad)]
                if v is None
            ]
            logger.warning(
                f"[tier1b] event {occ.event_id}: missing configs {missing}, skipping row"
            )
            stats.events_missing_config += 1
            continue

        if dim is None:
            dim = local.shape[0]

        rows.append((occ, local, medium, broad))

    if not rows:
        return

    stats.events_written += len(rows)

    concept_ids = [stable_hash(f"{o.doc_id}:{o.token_idx}") for o, *_ in rows]


    store.append_events(
        event_id=np.asarray([o.event_id for o, *_ in rows], dtype=np.int64),
        concept_id=np.asarray(concept_ids, dtype=np.int64),
        emb_local=np.stack([r[1] for r in rows]),
        emb_medium=np.stack([r[2] for r in rows]),
        emb_broad=np.stack([r[3] for r in rows]),
        vector_id=np.asarray([o.vector_id for o, *_ in rows], dtype=np.int64),
        doc_id=np.asarray([o.doc_id for o, *_ in rows], dtype="U32"),
        token_idx=np.asarray([o.token_idx for o, *_ in rows], dtype=np.int64),
        token=np.asarray([o.token for o, *_ in rows], dtype="U32"),
        # Carrying forward the ORIGINAL (medium-scale) window coordinates
        # for provenance - the store has one window_id/window_token_pos
        # pair per event, not one per config, so this necessarily refers
        # to the medium window specifically, same as Tier 1's own writes.
        window_id=np.asarray([o.orig_window_id for o, *_ in rows], dtype=np.int64),
        window_token_pos=np.asarray([o.orig_window_token_pos for o, *_ in rows], dtype=np.int32),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument(
        "--pooling-scope",
        choices=["mask_only", "context"],
        default="mask_only",
        help="mask_only: pool masked position(s) only. context: pool everything except the masked position(s).",
    )
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    process_concept(args.concept, pooling_scope=args.pooling_scope, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
