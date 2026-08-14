# retrieval/tests/unit/test_parquet_context_unit.py

"""
pytest src/retrieval/tests/unit/test_parquet_context_unit.py -v -s
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.models import SearchResult
from retrieval.parquet_context import ParquetContext


def write_corpus(
    root,
) -> None:
    table = pa.table(
        {
            "event_id": np.array(
                [101, 102, 103, 104, 105, 106, 107],
                dtype=np.uint64,
            ),
            "doc_id": [
                "DOC1",
                "DOC1",
                "DOC1",
                "DOC1",
                "DOC1",
                "DOC1",
                "DOC1",
            ],
            "token": [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
            ],
            "token_idx": np.array(
                [0, 1, 2, 3, 4, 5, 6],
                dtype=np.int64,
            ),
            "year": np.array(
                [1625, 1625, 1625, 1625, 1625, 1625, 1625],
                dtype=np.int16,
            ),
            "emb_local": [
                [0.0] * 768,
                [0.0] * 768,
                [0.0] * 768,
                [0.0] * 768,
                [0.0] * 768,
                [0.0] * 768,
                [0.0] * 768,
            ],
        }
    )

    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["year"],
    )


def test_get_excludes_centre_and_orders_context(
    tmp_path,
) -> None:
    write_corpus(tmp_path)

    context = ParquetContext(
        tmp_path,
        context_before=2,
        context_after=2,
    )

    result = context.get(
        104,
    )

    assert result.event_id == 104
    assert result.distance == 0.0

    assert result.observation["token"] == "three"
    assert result.observation["token_idx"] == 3

    assert [
        token.token
        for token in result.before
    ] == [
        "one",
        "two",
    ]

    assert [
        token.token
        for token in result.after
    ] == [
        "four",
        "five",
    ]

    assert all(
        token.event_id != result.event_id
        for token in (
            *result.before,
            *result.after,
        )
    )


def test_context_respects_document_boundaries(
    tmp_path,
) -> None:
    write_corpus(tmp_path)

    context = ParquetContext(
        tmp_path,
        context_before=10,
        context_after=10,
    )

    result = context.get(
        101,
    )

    assert [
        token.token
        for token in result.before
    ] == []

    assert [
        token.token
        for token in result.after
    ] == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
    ]


def test_zero_context_returns_only_centre(
    tmp_path,
) -> None:
    write_corpus(tmp_path)

    context = ParquetContext(
        tmp_path,
        context_before=0,
        context_after=0,
    )

    result = context.get(
        104,
    )

    assert result.before == ()
    assert result.after == ()
    assert result.text == "three"


def test_text_reconstructs_context_window(
    tmp_path,
) -> None:
    write_corpus(tmp_path)

    context = ParquetContext(
        tmp_path,
        context_before=2,
        context_after=2,
    )

    result = context.get(
        104,
    )

    assert result.text == (
        "one two three four five"
    )


def test_get_many_preserves_result_order_and_distances(
    tmp_path,
) -> None:
    write_corpus(tmp_path)

    context = ParquetContext(
        tmp_path,
        context_before=1,
        context_after=1,
    )

    search_result = SearchResult(
        event_ids=np.array(
            [106, 102, 104],
            dtype=np.uint64,
        ),
        distances=np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
    )

    results = context.get_many(
        search_result,
    )

    assert [
        result.event_id
        for result in results
    ] == [
        106,
        102,
        104,
    ]

    np.testing.assert_allclose(
        [
            result.distance
            for result in results
        ],
        np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
    )

    assert [
        result.text
        for result in results
    ] == [
        "four five six",
        "zero one two",
        "two three four",
    ]


def test_get_missing_observation_raises_key_error(
    tmp_path,
) -> None:
    write_corpus(tmp_path)

    context = ParquetContext(
        tmp_path,
    )

    try:
        context.get(999)
    except KeyError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError(
            "Expected missing observation to raise KeyError"
        )
