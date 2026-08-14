from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from retrieval.models import SearchResult, SearchSpace
from retrieval.observation_index_store import ObservationIndexStore
from retrieval.observation_retriever import IndexedObservationRetriever
from retrieval.parquet_context import ObservationContext


def _context(event_id: int, distance: float) -> ObservationContext:
    return ObservationContext(
        event_id=event_id,
        distance=distance,
        observation={
            "event_id": event_id,
            "corpus": "eebo",
            "doc_id": f"D{event_id}",
            "token_idx": event_id,
            "token": f"token-{event_id}",
        },
        before=(),
        after=(),
    )


def test_search_uses_index_for_requested_space() -> None:
    space = SearchSpace(year=1625, scale="local")

    index = Mock()
    index.search.return_value = SearchResult(
        event_ids=np.array([101, 102], dtype=np.uint64),
        distances=np.array([0.1, 0.2], dtype=np.float32),
    )

    index_store = Mock(spec=ObservationIndexStore)
    index_store.get.return_value = index

    context = Mock()
    context.get_many.return_value = [
        _context(101, 0.1),
        _context(102, 0.2),
    ]

    retriever = IndexedObservationRetriever(
        index_store=index_store,
        context=context,
    )

    query = np.ones(768, dtype=np.float32)

    result = retriever.search(
        query,
        space=space,
        k=2,
    )

    index_store.get.assert_called_once_with(space)
    index.search.assert_called_once_with(
        query,
        k=2,
    )
    context.get_many.assert_called_once_with(
        index.search.return_value,
    )

    assert result == [
        _context(101, 0.1),
        _context(102, 0.2),
    ]


def test_batch_search_uses_index_for_requested_space() -> None:
    space = SearchSpace(
        year=1625,
        scale="local",
    )

    index = Mock()
    index.batch_search.return_value = [
        SearchResult(
            event_ids=np.array([101, 102], dtype=np.uint64),
            distances=np.array([0.1, 0.2], dtype=np.float32),
        ),
        SearchResult(
            event_ids=np.array([103], dtype=np.uint64),
            distances=np.array([0.3], dtype=np.float32),
        ),
    ]

    index_store = Mock(spec=ObservationIndexStore)
    index_store.get.return_value = index

    context = Mock()
    context.get_many.side_effect = [
        [_context(101, 0.1), _context(102, 0.2)],
        [_context(103, 0.3)],
    ]

    retriever = IndexedObservationRetriever(
        index_store=index_store,
        context=context,
    )

    queries = np.ones((2, 768), dtype=np.float32)

    result = retriever.batch_search(
        queries,
        space=space,
        k=2,
    )

    index_store.get.assert_called_once_with(space)
    index.batch_search.assert_called_once_with(
        queries,
        k=2,
    )

    assert context.get_many.call_count == 2
    assert result == [
        [_context(101, 0.1), _context(102, 0.2)],
        [_context(103, 0.3)],
    ]