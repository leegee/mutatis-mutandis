#!/usr/bin/env python
"""
eebo_event_lookup.py

Event metadata resolution layer for EEBO semantic event embeddings.

ARCHITECTURAL ROLE
------------------

This module provides a stable mapping from FAISS event IDs
to historical linguistic provenance.

It is intentionally separated from:

    - FAISS (geometry / similarity space)
    - Zarr (embedding storage / event log)
    - Postgres (canonical source of record)

Once loaded, this structure is read-only.

DESIGN INTENT
-------------

FAISS returns:
    event_id → nearest neighbours in semantic space

This module resolves:
    event_id → linguistic and historical context

Together they form:
    geometric similarity + historical interpretation

CORE INVARIANT
---------------

event_id is a globally stable identifier corresponding to:
    one token occurrence in one document at one position
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from lib.corpus_db import get_connection
from lib.eebo_logging import logger


# ------------------------------------------------------------
# Event structure
# ------------------------------------------------------------

@dataclass(frozen=True)
class EventMetadata:
    """
    Immutable representation of a semantic event.
    """

    event_id: int
    token: str
    doc_id: str
    token_idx: int
    pub_year: int


# ------------------------------------------------------------
# Lookup store
# ------------------------------------------------------------

class EeboEventLookup:
    """
    In-memory event dictionary.

    Intended lifecycle:
        1. load() once at startup
        2. reused across FAISS queries
        3. never mutated

    This avoids repeated DB joins during analysis.
    """

    def __init__(self):
        self._events: Dict[int, EventMetadata] = {}

    def load(self) -> None:
        """
        Materialise event metadata from Postgres.

        Postgres remains canonical source of truth,
        but is NOT queried during analysis.
        """

        logger.info("[event_lookup] loading event metadata")

        conn = get_connection()

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    t.vector_id,
                    t.token,
                    t.doc_id,
                    t.token_idx,
                    d.pub_year
                FROM pamphlet_tokens t
                JOIN pamphlet_corpus d
                    ON d.doc_id = t.doc_id
            """)

            count = 0

            for event_id, token, doc_id, token_idx, pub_year in cur:
                self._events[int(event_id)] = EventMetadata(
                    event_id=int(event_id),
                    token=str(token),
                    doc_id=str(doc_id),
                    token_idx=int(token_idx),
                    pub_year=int(pub_year),
                )
                count += 1

        conn.close()

        logger.info(f"[event_lookup] loaded events={count}")

    # --------------------------------------------------------
    # Access methods
    # --------------------------------------------------------

    def get(self, event_id: int) -> EventMetadata:
        """
        Resolve a single event ID.
        """

        try:
            return self._events[int(event_id)]
        except KeyError:
            raise KeyError(f"Unknown event_id={event_id}")

    def batch(self, event_ids: Iterable[int]) -> list[EventMetadata]:
        """
        Resolve multiple event IDs efficiently.
        """

        return [self.get(eid) for eid in event_ids]

    def __len__(self) -> int:
        return len(self._events)
