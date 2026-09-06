# tests/test_diachronic_search.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import (
    SearchResult,
    SearchSpace,
)


def test_search_space_buckets_forward() -> None:
    space = SearchSpace(
        years=(1500, 1699),
        scale=("medium",),
    )

    assert list(
        space.buckets(
            50,
            direction="forward",
        )
    ) == [
        (1500, 1549),
        (1550, 1599),
        (1600, 1649),
        (1650, 1699),
    ]


def test_search_space_buckets_backward() -> None:
    space = SearchSpace(
        years=(1500, 1699),
        scale=("medium",),
    )

    assert list(
        space.buckets(
            50,
            direction="backward",
        )
    ) == [
        (1650, 1699),
        (1600, 1649),
        (1550, 1599),
        (1500, 1549),
    ]


def test_search_space_buckets_partial_final_bucket() -> None:
    space = SearchSpace(
        years=(1523, 1601),
        scale=("medium",),
    )

    assert list(
        space.buckets(
            50,
            direction="forward",
        )
    ) == [
        (1523, 1572),
        (1573, 1601),
    ]


def test_search_space_buckets_requires_bounded_years() -> None:
    space = SearchSpace(
        years=None,
        scale=("medium",),
    )

    with pytest.raises(ValueError):
        list(space.buckets(50))


def test_search_space_buckets_rejects_invalid_size() -> None:
    space = SearchSpace(
        years=(1500, 1599),
        scale=("medium",),
    )

    with pytest.raises(ValueError):
        list(space.buckets(0))


def test_search_space_buckets_rejects_invalid_direction() -> None:
    space = SearchSpace(
        years=(1500, 1599),
        scale=("medium",),
    )

    with pytest.raises(ValueError):
        list(
            space.buckets(
                50,
                direction="sideways",
            )
        )


@dataclass
class FakeTable:
    name: str


def make_store(
    buckets: list[tuple[str, str, int, int]],
) -> LanceObservationIndexStore:
    store = object.__new__(LanceObservationIndexStore)

    store._available_scales = (
        "local",
        "medium",
        "broad",
    )

    store._tables = {
        (
            scale,
            model,
            year_start,
            year_end,
        ): FakeTable(
            f"{scale}:{model}:{year_start}-{year_end}"
        )
        for scale, model, year_start, year_end in buckets
    }

    return store


def test_store_resolves_physical_buckets_forward() -> None:
    store = make_store([
        ("medium", "macberth", 1450, 1499),
        ("medium", "macberth", 1500, 1549),
        ("medium", "macberth", 1550, 1599),
        ("medium", "macberth", 1600, 1649),
    ])

    space = SearchSpace(
        years=(1476, 1624),
        scale=("medium",),
    )

    assert store._buckets_for_search(
        space,
        direction="forward",
    ) == (
        (1450, 1499),
        (1500, 1549),
        (1550, 1599),
        (1600, 1649),
    )


def test_store_resolves_physical_buckets_backward() -> None:
    store = make_store([
        ("medium", "macberth", 1450, 1499),
        ("medium", "macberth", 1500, 1549),
        ("medium", "macberth", 1550, 1599),
        ("medium", "macberth", 1600, 1649),
    ])

    space = SearchSpace(
        years=(1476, 1624),
        scale=("medium",),
    )

    assert store._buckets_for_search(
        space,
        direction="backward",
    ) == (
        (1600, 1649),
        (1550, 1599),
        (1500, 1549),
        (1450, 1499),
    )


def test_store_excludes_non_overlapping_physical_buckets() -> None:
    store = make_store([
        ("medium", "macberth", 1400, 1449),
        ("medium", "macberth", 1450, 1499),
        ("medium", "macberth", 1500, 1549),
        ("medium", "macberth", 1550, 1599),
        ("medium", "macberth", 1600, 1649),
        ("medium", "macberth", 1650, 1699),
    ])

    space = SearchSpace(
        years=(1501, 1598),
        scale=("medium",),
    )

    assert store._buckets_for_search(
        space,
        direction="forward",
    ) == (
        (1500, 1549),
        (1550, 1599),
    )


class FakeIndex:
    def __init__(
        self,
        scale: str,
        bucket: tuple[int, int],
        calls: list,
    ) -> None:
        self.scale = scale
        self.bucket = bucket
        self.calls = calls

    def search(
        self,
        query,
        *,
        k: int,
    ) -> SearchResult:
        self.calls.append(
            (
                self.scale,
                self.bucket,
                np.asarray(query).copy(),
                k,
            )
        )

        return SearchResult(
            event_ids=np.asarray(
                [self.bucket[0]],
                dtype=np.uint64,
            ),
            distances=np.asarray(
                [1.0],
                dtype=np.float32,
            ),
        )


def test_diachronic_search_visits_buckets_in_requested_direction(
    monkeypatch,
) -> None:
    store = make_store([
        ("local", "macberth", 1500, 1549),
        ("local", "macberth", 1550, 1599),
        ("medium", "macberth", 1500, 1549),
        ("medium", "macberth", 1550, 1599),
    ])

    calls = []

    def fake_get(space):
        assert space.years is not None

        bucket = space.years

        return {
            scale: FakeIndex(
                scale,
                bucket,
                calls,
            )
            for scale in space.scale
        }

    monkeypatch.setattr(
        store,
        "get",
        fake_get,
    )

    queries = {
        "local": np.ones(
            4,
            dtype=np.float32,
        ),
        "medium": np.full(
            4,
            2.0,
            dtype=np.float32,
        ),
    }

    space = SearchSpace(
        years=(1501, 1598),
        scale=("local", "medium"),
    )

    results = list(
        store.diachronic_search(
            queries,
            space,
            k=60,
            direction="backward",
        )
    )

    assert [
        bucket
        for bucket, _ in results
    ] == [
        (1550, 1599),
        (1500, 1549),
    ]

    assert [
        (scale, bucket)
        for scale, bucket, _, _ in calls
    ] == [
        ("local", (1550, 1599)),
        ("medium", (1550, 1599)),
        ("local", (1500, 1549)),
        ("medium", (1500, 1549)),
    ]


def test_diachronic_search_keeps_query_vectors_separate_by_scale(
    monkeypatch,
) -> None:
    store = make_store([
        ("local", "macberth", 1500, 1549),
        ("medium", "macberth", 1500, 1549),
    ])

    calls = []

    def fake_get(space):
        bucket = space.years

        return {
            scale: FakeIndex(
                scale,
                bucket,
                calls,
            )
            for scale in space.scale
        }

    monkeypatch.setattr(
        store,
        "get",
        fake_get,
    )

    local_query = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )

    medium_query = np.asarray(
        [10.0, 20.0, 30.0, 40.0],
        dtype=np.float32,
    )

    list(
        store.diachronic_search(
            {
                "local": local_query,
                "medium": medium_query,
            },
            SearchSpace(
                years=(1500, 1549),
                scale=("local", "medium"),
            ),
            k=10,
        )
    )

    assert np.array_equal(
        calls[0][2],
        local_query,
    )

    assert np.array_equal(
        calls[1][2],
        medium_query,
    )


def test_diachronic_search_rejects_missing_scale_query() -> None:
    store = make_store([
        ("local", "macberth", 1500, 1549),
        ("medium", "macberth", 1500, 1549),
    ])

    space = SearchSpace(
        years=(1500, 1549),
        scale=("local", "medium"),
    )

    with pytest.raises(ValueError):
        list(
            store.diachronic_search(
                {
                    "local": np.ones(
                        4,
                        dtype=np.float32,
                    ),
                },
                space,
                k=10,
            )
        )