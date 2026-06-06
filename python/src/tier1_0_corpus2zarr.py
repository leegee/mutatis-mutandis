#!/usr/bin/env python
"""
tier1_corpus2zarr.py

Contextual Observation Log (Tier 1)

Core invariant
--------------
- Postgres defines corpus identity
- concept_id defines stable lexical occurrence identity
- event_id defines contextual embedding observation identity
- Later, FAISS indexes event_id space ONLY

Check the minimal STOPWORDS list

"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import xxhash
import unicodedata

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, SLICES, EMBED_BATCH_SIZE
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.macberth import load_macberth


WINDOW_SIZE = 512
STRIDE = WINDOW_SIZE // 2

STOPWORDS = {
    # articles
    "the", "a", "an",
    # pronouns
    "i", "he", "she", "it", "we", "they", "who", "which", "that",
    "his", "her", "its", "our", "their", "my", "thy", "your", "hast",
    "art", "is", "who", "thine", "mine", "his", "her", "its",
    "him", "them", "us", "me", "thee", "thy", "thou", "thoust",
    # conjunctions / prepositions
    "and", "or", "but", "of", "in", "to", "for", "with", "by",
    "at", "from", "as", "on", "into", "upon", "unto", "not",
    "nor", "yet", "so", "if", "be", "is", "are", "was", "were", "where",
    "have", "hath", "hath", "do", "doth",
    "shall", "will", "may",  "should", "would", "could", # "might",
    "than", "then", "when",
    "this", "these", "those", "all", "no", "any", "such", "many", "some",
}


def stable_hash(key: str) -> np.int64:
    h = xxhash.xxh64(key, seed=0).intdigest()
    return np.int64(h & 0x7FFFFFFFFFFFFFFF)


def is_content_token(token: str) -> bool:
    """Reject punctuation, whitespace, stopwords, and bare symbols.
    Early modern texts include many non-lexical characters."""
    stripped = token.strip().lower()
    if not stripped or stripped in STOPWORDS:
        return False
    if all(unicodedata.category(c).startswith(("P", "S", "Z")) for c in stripped):
        return False
    return True


#
# TODO Extrapolate
#
@dataclass(slots=True)
class Event:
    event_id:         np.int64    # unique contextual observation
    concept_id:       np.int64    # stable corpus token identity
    doc_id:           str
    token_idx:        int         # original corpus token position
    window_start:     int         # contextual frame origin
    window_token_pos: int         # intra-window transformer position
    token:            str
    vector_id:        int
    vec:              np.ndarray

    @staticmethod
    def make(doc_id, corpus_token_idx, window_start, window_token_pos, token, vector_id, vec):
        concept_id = stable_hash(f"{doc_id}:{corpus_token_idx}")
        event_id   = stable_hash(f"{doc_id}:{corpus_token_idx}:{window_start}:{window_token_pos}")
        return Event(
            event_id         = event_id,
            concept_id       = concept_id,
            doc_id           = doc_id,
            token_idx        = corpus_token_idx,
            window_start     = window_start,
            window_token_pos = window_token_pos,
            token            = token,
            vector_id        = vector_id,
            vec              = vec.astype(np.float32),
        )



@dataclass
class DocBuffer:
    """Accumulates content tokens for one document before embedding."""
    doc_id:     str
    tokens:     list = None
    vector_ids: list = None
    token_idxs: list = None  # original corpus positions

    def __post_init__(self):
        self.tokens     = []
        self.vector_ids = []
        self.token_idxs = []

    def append(self, token, vector_id, token_idx):
        self.tokens.append(token)
        self.vector_ids.append(vector_id)
        self.token_idxs.append(token_idx)

    def __bool__(self):
        return bool(self.tokens)



class EmbeddingPipeline:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device

    def embed_doc(self, buf: DocBuffer) -> list[Event]:
        """Encode, window, batch-forward, and extract events for one document."""
        input_ids, attention_mask, word_ids = self._encode(buf.tokens)

        events = []
        batch  = []

        for window_start, ids, mask, wids in self._iter_windows(input_ids, attention_mask, word_ids):
            batch.append({
                "input_ids":    ids,
                "mask":         mask,
                "word_ids":     wids,
                "window_start": window_start,
            })
            if len(batch) >= EMBED_BATCH_SIZE:
                events.extend(self._flush_batch(buf, batch))
                batch.clear()

        if batch:
            events.extend(self._flush_batch(buf, batch))

        return events

    def _encode(self, tokens):
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )
        word_ids = enc.word_ids() or [None] * len(enc["input_ids"][0])
        return enc["input_ids"][0].tolist(), enc["attention_mask"][0].tolist(), word_ids

    @staticmethod
    def _iter_windows(input_ids, attention_mask, word_ids):
        n, start = len(input_ids), 0
        while start < n:
            end = min(start + WINDOW_SIZE, n)
            yield start, input_ids[start:end], attention_mask[start:end], word_ids[start:end]
            if end == n:
                break
            start += STRIDE

    def _flush_batch(self, buf: DocBuffer, batch: list) -> list[Event]:
        hidden_states = self._forward(batch)
        events = []
        for item, hidden in zip(batch, hidden_states):
            events.extend(self._extract_events(buf, item, hidden))
        return events

    def _forward(self, batch: list) -> np.ndarray:
        max_len = max(len(x["input_ids"]) for x in batch)

        def pad(seq):
            return seq + [0] * (max_len - len(seq))

        input_ids = torch.tensor(
            [pad(x["input_ids"]) for x in batch], dtype=torch.long
        ).to(self.device)

        mask = torch.tensor(
            [pad(x["mask"]) for x in batch], dtype=torch.long
        ).to(self.device)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=mask, return_dict=True)

        return out.last_hidden_state.cpu().numpy()

    @staticmethod
    def _extract_events(buf: DocBuffer, item: dict, hidden: np.ndarray) -> list[Event]:
        window_start = item["window_start"]
        return [
            Event.make(
                doc_id           = buf.doc_id,
                corpus_token_idx = buf.token_idxs[wid],
                window_start     = window_start,
                window_token_pos = i,
                token            = buf.tokens[wid],
                vector_id        = buf.vector_ids[wid],
                vec              = hidden[i],
            )
            for i, wid in enumerate(item["word_ids"])
            if wid is not None and wid >= 0
        ]



class SliceProcessor:
    def __init__(self, conn, pipeline: EmbeddingPipeline):
        self.conn     = conn
        self.pipeline = pipeline

    def process(self, slice_range):
        slice_id = f"{slice_range[0]}-{slice_range[1]}"
        logger.info(f"[SLICE START] {slice_id}")

        store = ZarrEmbeddingObservationStore(
            path=str(ZARR_ROOT / "tier1" / slice_id),
            dim=self.pipeline.model.config.hidden_size,
        )

        cur = self.conn.cursor(name=f"tier1_{slice_id}")
        cur.itersize = 10000
        cur.execute("""
            SELECT doc_id, token_idx, vector_id, token
            FROM pamphlet_tokens
            WHERE pub_year BETWEEN %s AND %s
            ORDER BY doc_id, token_idx
        """, slice_range)

        buf = None

        for doc_id, token_idx, vid, token in cur:
            if buf is None or doc_id != buf.doc_id:
                if buf:
                    self._flush(buf, store)
                buf = DocBuffer(doc_id=doc_id)

            if is_content_token(token):
                buf.append(token, vid, token_idx)

        if buf:
            self._flush(buf, store)

        logger.info(f"[SLICE COMPLETE] {slice_id}")

    def _flush(self, buf: DocBuffer, store: ZarrEmbeddingObservationStore):
        if not buf:
            return

        events = self.pipeline.embed_doc(buf)
        if not events:
            return

        (event_ids, concept_ids, doc_ids, token_idxs,
         window_starts, window_token_pos, tokens, vector_ids, vecs) = zip(*[
            (e.event_id, e.concept_id, e.doc_id, e.token_idx,
             e.window_start, e.window_token_pos, e.token, e.vector_id, e.vec)
            for e in events
        ])

        store.append_events(
            event_id         = np.asarray(event_ids,        dtype=np.int64),
            concept_id       = np.asarray(concept_ids,      dtype=np.int64),
            emb_raw          = np.stack(vecs),
            vector_id        = np.asarray(vector_ids,       dtype=np.int64),
            doc_id           = np.asarray(doc_ids,          dtype="U32"),
            token_idx        = np.asarray(token_idxs,       dtype=np.int32),
            window_id        = np.asarray(window_starts,    dtype=np.int32),
            window_token_pos = np.asarray(window_token_pos, dtype=np.int32),
            token            = np.asarray(tokens,           dtype=object),
        )



def clear_output_dir():
    path = ZARR_ROOT / "tier1"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-clear", action="store_true")
    p.add_argument("--first",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_clear:
        logger.info("Clearing Tier 1 output")
        clear_output_dir()

    conn     = get_connection()
    mac      = load_macberth()
    pipeline = EmbeddingPipeline(mac.tokenizer, mac.model, mac.device)
    proc     = SliceProcessor(conn, pipeline)

    slices = SLICES[:1] if args.first else SLICES

    for s in slices:
        proc.process(s)

    conn.close()


if __name__ == "__main__":
    main()
