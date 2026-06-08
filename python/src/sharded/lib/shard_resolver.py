"""
lib/shard_resolver.py

Maps (corpus_id, pub_year, WindowStrategy) to a canonical Zarr shard path.

Physical layout on disk:

    <ZARR_ROOT>/
        meta/                          ← lookup tables (place, author, model, view)
        EEBO/
            1600-1649/
                MacBERTh/
                    doc/
                    paragraph/
                    sentence/
                    sliding_512_256/
            1650-1699/
                MacBERTh/
                    ...
            undated/
                MacBERTh/
                    ...
        ECCO/
            ...
        Periodicals/
            ...
        DBpedia/
            ...

Design notes
------------
- PERIOD_SIZE is the only constant that controls shard granularity.
  Change it once here; all paths recalculate.
- model_name is kept separate from WindowStrategy so that swapping
  models (e.g. MacBERTh → SBERT) produces a new branch without
  touching the strategy layer.
- resolve() creates the directory if absent — safe to call repeatedly.
- all_shards() lets the search layer enumerate shards without
  hardcoding paths.
"""

from __future__ import annotations

from pathlib import Path

from lib.eebo_config import ZARR_ROOT
from lib.window_strategy import WindowStrategy, WINDOW_STRATEGIES


PERIOD_SIZE: int = 50   # years per temporal shard

KNOWN_CORPORA: frozenset[str] = frozenset({
    "EEBO",
    "ECCO",
    "Periodicals",
    "DBpedia",
})


class ShardResolver:

    def __init__(self, model_name: str = "MacBERTh"):
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def resolve(
        self,
        corpus_id: str,
        pub_year:  int | None,
        strategy:  WindowStrategy,
    ) -> Path:
        """
        Return the canonical shard path for this (corpus, period, strategy)
        combination, creating the directory if it does not yet exist.
        """
        self._validate_corpus(corpus_id)
        path = (
            ZARR_ROOT
            / corpus_id
            / self._period_label(pub_year)
            / self.model_name
            / strategy.tag
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_meta(self) -> Path:
        """
        Return the path for the /meta group (lookup tables).
        Created on first call.
        """
        path = ZARR_ROOT / "meta"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def all_shards(
        self,
        corpus_id: str | None = None,
        strategy:  WindowStrategy | None = None,
    ) -> list[Path]:
        """
        Return all existing shard paths, optionally filtered by corpus
        and/or strategy.  Useful for search-layer enumeration and for
        building FAISS index lists.
        """
        corpora = [corpus_id] if corpus_id else list(KNOWN_CORPORA)
        tag     = strategy.tag if strategy else "*"
        return sorted(
            p
            for corpus in corpora
            for p in (ZARR_ROOT / corpus).glob(
                f"*/{self.model_name}/{tag}"
            )
            if p.is_dir()
        )

    def all_strategies_for_shard(
        self,
        corpus_id: str,
        pub_year:  int | None,
    ) -> list[Path]:
        """
        Return existing strategy paths under a given (corpus, period) node.
        """
        self._validate_corpus(corpus_id)
        base = ZARR_ROOT / corpus_id / self._period_label(pub_year) / self.model_name
        return sorted(p for p in base.glob("*") if p.is_dir())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _period_label(pub_year: int | None) -> str:
        if pub_year is None:
            return "undated"
        base = (pub_year // PERIOD_SIZE) * PERIOD_SIZE
        return f"{base}-{base + PERIOD_SIZE - 1}"

    @staticmethod
    def _validate_corpus(corpus_id: str) -> None:
        if corpus_id not in KNOWN_CORPORA:
            raise ValueError(
                f"Unknown corpus {corpus_id!r}. "
                f"Expected one of: {sorted(KNOWN_CORPORA)}"
            )
