"""
tier2/test_memory.py

Claude tests for the memory-conservation changes made to tier2:

1. Streaming/chunked concept processing keeps any single FAISS search
   call bounded to `batch_size` positions, regardless of how many
   events a concept matches in total (analysis.iter_concept_batches).

2. Per-year FAISS index eviction: each year's indices are loaded at
   most once for a whole batch of concepts, and evicted exactly once,
   right after the last concept in the batch that needs them
   (analysis.build_eviction_schedule + resources.LazyYearIndices).

3. Correctness: concept_field_events.role stays 'seed' for every one
   of a concept's own seed events, even when a seed event is also a
   FAISS neighbour of another seed in the same concept — this must not
   depend on batch/write order (persistence.write_concept_batch).

Run with:  pytest test_tier2_memory.py -v
       or: python test_tier2_memory.py

This test stubs out every `lib.*` module tier2 imports (FAISS,
Postgres, Zarr, logging, config, concept resolution) so it can run
standalone against a real (temp-file) SQLite database, without the
actual EEBO corpus, FAISS indices, or a Postgres connection.
"""

from __future__ import annotations

import os
import sys
import types
import sqlite3
import tempfile

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Stub out lib.* before tier2 is ever imported, so tier2's own imports
# (from lib.eebo_faiss import ..., etc.) resolve to these fakes.
# ----------------------------------------------------------------------

def _install_lib_stubs(load_calls, evict_log, search_calls):
    lib_pkg = types.ModuleType("lib")
    sys.modules["lib"] = lib_pkg

    # --- lib.eebo_faiss ---
    faiss_mod = types.ModuleType("lib.eebo_faiss")

    class FakeIndex:
        ntotal = 5

    class FakeEeboFaissIndex:
        @staticmethod
        def load(path):
            load_calls.append(path)
            return FakeIndex()

    def fake_multiscale_search(indexes, lookup, positions, top_n, pub_year, rrf_k, oversample):
        # Force the index for this year to actually load, same as the
        # real multiscale_search would by touching indexes[pub_year].
        _ = indexes[pub_year]
        search_calls.append((pub_year, len(positions)))

        n = len(lookup.event_id)
        # Deterministic fake neighbour: "the next event in the array".
        # This is deliberately adversarial for the seed/neighbour role
        # test below — chosen so neighbours very often collide with
        # other seed events of the same concept.
        return [
            [{"event_id": int((p + 1) % n), "rrf_score": 1.0}]
            for p in positions
        ]

    faiss_mod.EeboFaissIndex = FakeEeboFaissIndex
    faiss_mod.multiscale_search = fake_multiscale_search
    sys.modules["lib.eebo_faiss"] = faiss_mod

    # --- lib.eebo_logging ---
    log_mod = types.ModuleType("lib.eebo_logging")

    class FakeLogger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    fake_logger = FakeLogger()
    log_mod.logger = fake_logger

    def setEmit(emit, prefix, ctx):
        return fake_logger

    log_mod.setEmit = setEmit
    sys.modules["lib.eebo_logging"] = log_mod

    # --- lib.eebo_db (Postgres) ---
    db_mod = types.ModuleType("lib.eebo_db")

    def get_connection():
        raise RuntimeError("no postgres available in test")

    db_mod.get_connection = get_connection
    sys.modules["lib.eebo_db"] = db_mod

    # --- lib.eebo_config ---
    cfg_mod = types.ModuleType("lib.eebo_config")
    cfg_mod.CORPUS_TIER2_DB_PATH = None
    cfg_mod.CORPUS_TIER2_MASKED_DB_PATH = None
    cfg_mod.ZARR_PATH = None
    cfg_mod.MASKED_ZARR_PATH = None
    cfg_mod.faiss_index_paths = lambda **k: {}
    cfg_mod.discover_index_years = lambda *a, **k: []
    sys.modules["lib.eebo_config"] = cfg_mod

    # --- lib.zarr_event_lookup ---
    zarr_mod = types.ModuleType("lib.zarr_event_lookup")

    class ZarrEventLookup:
        pass

    zarr_mod.ZarrEventLookup = ZarrEventLookup
    sys.modules["lib.zarr_event_lookup"] = zarr_mod

    # --- lib.concept_resolve ---
    cr_mod = types.ModuleType("lib.concept_resolve")
    cr_mod.resolve_concepts = lambda **k: []
    sys.modules["lib.concept_resolve"] = cr_mod

    # --- lib.get_processed_concepts ---
    gpc_mod = types.ModuleType("lib.get_processed_concepts")
    gpc_mod.get_processed_concepts = lambda *a, **k: set()
    sys.modules["lib.get_processed_concepts"] = gpc_mod


_LOAD_CALLS = []
_EVICT_LOG = []
_SEARCH_CALLS = []
_install_lib_stubs(_LOAD_CALLS, _EVICT_LOG, _SEARCH_CALLS)

# tier2 lives one directory up from this test file (adjust if you move it).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tier2.resources import LazyYearIndices  # noqa: E402

_orig_evict = LazyYearIndices.evict


def _logged_evict(self, year):
    _EVICT_LOG.append(year)
    _orig_evict(self, year)


LazyYearIndices.evict = _logged_evict

from tier2.orchestrator import service  # noqa: E402


# ----------------------------------------------------------------------
# Fake lookup: N events, positions == event_ids, publication years
# cycling through a fixed range so a concept's matches necessarily
# span multiple years (forcing the by-year grouping / chunking / index
# loading logic to actually exercise itself).
# ----------------------------------------------------------------------

class FakeLookup:
    def __init__(self, n, years, forms_map):
        self.event_id = np.arange(n)
        self.vector_id = np.arange(n)
        self.token = np.array([f"tok{i}" for i in range(n)])
        self.doc_id = np.array([f"doc{i % 37}" for i in range(n)])
        self.pub_year = np.array([years[i % len(years)] for i in range(n)])
        self.token_idx = np.arange(n)
        self.window_id = np.zeros(n, dtype=int)
        self.window_token_pos = np.full(n, -1)
        self._forms_map = forms_map

    def find_matching_event_ids(self, forms, false_positives):
        forms = {f.upper() for f in forms}
        for key, ids in self._forms_map.items():
            if key in forms:
                return list(ids)
        return []

    def get_pos(self, eid):
        return eid

    def get_event(self, eid):
        eid = int(eid)
        return {
            "vector_id": int(self.vector_id[eid]),
            "token": str(self.token[eid]),
            "doc_id": str(self.doc_id[eid]),
            "pub_year": int(self.pub_year[eid]),
            "token_idx": int(self.token_idx[eid]),
            "window_id": int(self.window_id[eid]),
            "window_token_pos": int(self.window_token_pos[eid]),
        }

    def attach_index(self, indexes):
        pass


class FakePath:
    """Minimal stand-in for pathlib.Path, as expected by
    tier2.persistence.initialise_database."""

    def __init__(self, p):
        self._p = p

    def exists(self):
        return os.path.exists(self._p)

    def unlink(self):
        os.remove(self._p)

    def __fspath__(self):
        return self._p

    def __str__(self):
        return self._p


YEARS = [1640, 1641, 1642, 1643, 1644]
N_LARGE = 25000   # a "KING"-sized concept: bigger than BATCH_SIZE
N_SMALL = 50      # a small concept, for contrast
N = N_LARGE + N_SMALL


@pytest.fixture
def env():
    """Fresh lookup + indexes + temp SQLite db + call logs for one test."""
    _LOAD_CALLS.clear()
    _EVICT_LOG.clear()
    _SEARCH_CALLS.clear()

    lookup = FakeLookup(
        n=N,
        years=YEARS,
        forms_map={
            "KING": range(N_LARGE),
            "SMALL": range(N_LARGE, N_LARGE + N_SMALL),
        },
    )

    paths_by_year = {
        y: {"local": f"{y}-l", "medium": f"{y}-m", "broad": f"{y}-b"}
        for y in YEARS
    }
    indexes = LazyYearIndices(paths_by_year, workers=1)

    tmpdir = tempfile.mkdtemp()
    db_path = FakePath(os.path.join(tmpdir, "test.db"))

    return {
        "lookup": lookup,
        "indexes": indexes,
        "db_path": db_path,
    }


def _run(env, concepts_to_run, batch_size=2000):
    return service(
        lookup=env["lookup"],
        indexes=env["indexes"],
        concepts_to_run=concepts_to_run,
        db_path=env["db_path"],
        clear=True,
        batch_size=batch_size,
    )


# ----------------------------------------------------------------------
# 1. Chunking keeps any single FAISS search call bounded.
# ----------------------------------------------------------------------

def test_chunking_bounds_search_batch_size(env):
    concepts_to_run = [
        ("KING", {"forms": ["king"]}),
        ("SMALL", {"forms": ["small"]}),
    ]

    batch_size = 2000
    out = _run(env, concepts_to_run, batch_size=batch_size)

    assert out["summary"]["concepts_written"] == 2
    assert out["summary"]["concepts_empty"] == []

    max_batch = max(n for _, n in _SEARCH_CALLS)
    assert max_batch <= batch_size, (
        f"a single FAISS search call processed {max_batch} positions, "
        f"exceeding batch_size={batch_size} — chunking is not bounding "
        f"peak memory as intended"
    )

    # KING (25,000 events) must have required more than one search call;
    # if it didn't, chunking isn't actually happening for large concepts.
    assert len(_SEARCH_CALLS) > 1


# ----------------------------------------------------------------------
# 2. Each year's FAISS indices load at most once and get evicted
#    exactly once, right after the batch's last user of that year.
# ----------------------------------------------------------------------

def test_eviction_schedule_loads_and_evicts_each_year_once(env):
    concepts_to_run = [
        ("KING", {"forms": ["king"]}),
        ("SMALL", {"forms": ["small"]}),
    ]

    _run(env, concepts_to_run, batch_size=2000)

    # 5 years * 3 scales (local/medium/broad) = 15 load calls total,
    # regardless of how many chunks/concepts touched each year.
    assert len(_LOAD_CALLS) == len(YEARS) * 3, (
        f"expected each year's 3 scales to load exactly once "
        f"({len(YEARS) * 3} total), got {len(_LOAD_CALLS)} — a year is "
        f"being reloaded, which means eviction is happening too early"
    )

    assert sorted(_EVICT_LOG) == sorted(YEARS), (
        "expected every year to be evicted exactly once by the end of "
        "the batch"
    )

    # Nothing should still be resident once the whole batch is done.
    assert env["indexes"]._loaded == {}, (
        "FAISS indices are still resident in memory after the batch "
        "finished — eviction did not fire for every loaded year"
    )


# ----------------------------------------------------------------------
# 3. Seed/neighbour role correctness: a concept's own seed events must
#    never end up mislabeled 'neighbour', regardless of batch order.
# ----------------------------------------------------------------------

def test_seed_events_never_mislabeled_as_neighbours(env):
    concepts_to_run = [
        ("KING", {"forms": ["king"]}),
        ("SMALL", {"forms": ["small"]}),
    ]

    _run(env, concepts_to_run, batch_size=2000)

    con = sqlite3.connect(env["db_path"]._p)
    try:
        n_events_king = con.execute(
            "SELECT n_events FROM concepts WHERE concept='KING'"
        ).fetchone()[0]

        n_seed_king = con.execute(
            "SELECT COUNT(*) FROM concept_field_events "
            "WHERE concept='KING' AND role='seed'"
        ).fetchone()[0]

        # Every event_id should appear at most once per concept in
        # concept_field_events — if role assignment depended on write
        # order, this is where a seed would silently get overwritten
        # with role='neighbour' instead of being missing entirely.
        assert n_events_king == N_LARGE
        assert n_seed_king == N_LARGE, (
            f"expected all {N_LARGE} KING seed events to have "
            f"role='seed', but only {n_seed_king} do — some seeds were "
            f"overwritten with role='neighbour' by a later batch"
        )

        dupes = con.execute(
            "SELECT event_id FROM concept_field_events "
            "WHERE concept='KING' GROUP BY event_id HAVING COUNT(*) > 1"
        ).fetchall()
        assert dupes == [], f"duplicate concept_field_events rows: {dupes}"
    finally:
        con.close()


# ----------------------------------------------------------------------
# 4. build_eviction_schedule must not retain every concept's full
#    positions/event_ids for the duration of the batch — only a small
#    per-concept set of years touched. This is the "processing multiple
#    concepts at once" memory bug: an earlier version returned a
#    resolved_by_concept dict holding each concept's whole matched-event
#    payload simultaneously, for concepts that could be huge.
# ----------------------------------------------------------------------

def test_eviction_schedule_does_not_retain_full_concept_data(env):
    from tier2.analysis import build_eviction_schedule

    concepts_to_run = [
        ("KING", {"forms": ["king"]}),
        ("SMALL", {"forms": ["small"]}),
    ]

    years_by_concept, last_use = build_eviction_schedule(
        lookup=env["lookup"],
        concepts_to_run=concepts_to_run,
        false_positives=None,
    )

    # The schedule should carry only the years each concept touches —
    # a handful of ints per concept — never the underlying positions,
    # event_ids, or per-year position lists (which for a large concept
    # like KING would be tens of thousands of entries).
    for name, years in years_by_concept.items():
        assert isinstance(years, set)
        assert all(isinstance(y, int) for y in years)
        assert len(years) <= len(YEARS)

    assert years_by_concept["KING"] == set(YEARS)
    assert last_use == {y: 1 for y in YEARS}, (
        "SMALL (index 1) also touches every year and runs after KING, "
        "so it should be the last user of every year in this fixture"
    )

    # Nothing resembling a full per-concept payload should be reachable
    # off the returned objects.
    for name, years in years_by_concept.items():
        assert not hasattr(years, "keys"), (
            f"years_by_concept['{name}'] looks like a dict/mapping, not "
            f"a plain set of years — the schedule pass may be retaining "
            f"more than it should again"
        )


def test_batch_writes_do_not_depend_on_a_retained_resolved_dict(env):
    """
    Regression test for the specific fix in this turn: service() must
    not need to keep a concept's `resolved` dict alive across its own
    iter_concept_batches() call — each yielded batch carries its own
    `seed_ids` reference, and iter_concept_batches recomputes positions
    internally (resolved=None) rather than the caller pre-computing and
    holding them for every concept in the batch up front.
    """
    from tier2.analysis import iter_concept_batches

    kinds_seen = []
    for item in iter_concept_batches(
        concept_name="KING",
        concept={"forms": ["king"]},
        lookup=env["lookup"],
        indexes=env["indexes"],
        false_positives=None,
        batch_size=2000,
        # resolved intentionally omitted (defaults to None)
    ):
        kinds_seen.append(item["type"])
        if item["type"] == "batch":
            assert "seed_ids" in item, (
                "batch items must carry their own seed_ids so callers "
                "don't need to keep a separate resolved dict alive"
            )
            assert isinstance(item["seed_ids"], set)

    assert "batch" in kinds_seen
    assert kinds_seen[-1] == "final"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
