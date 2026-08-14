# retrieval/tests/unit/test_parquet_observations_unit.py
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.parquet_observations import ParquetObservationStore


def write_test_corpus(root) -> None:
    table = pa.table(
        {
            "event_id": pa.array(
                [
                    1001,
                    1002,
                    1003,
                ],
                type=pa.uint64(),
            ),
            "year": pa.array(
                [1625, 1625, 1625],
                type=pa.int32(),
            ),
            "document_id": [
                "doc-a",
                "doc-b",
                "doc-c",
            ],
            "token": [
                "white",
                "pale",
                "fair",
            ],
            "position": pa.array(
                [10, 20, 30],
                type=pa.int32(),
            ),
            "emb_local": [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            "emb_medium": [
                [0.11, 0.21],
                [0.31, 0.41],
                [0.51, 0.61],
            ],
        }
    )

    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["year"],
    )


def test_observation_store_exposes_non_embedding_columns(
    tmp_path,
) -> None:
    write_test_corpus(tmp_path)

    store = ParquetObservationStore(tmp_path)

    assert "event_id" in store.columns
    assert "document_id" in store.columns
    assert "token" in store.columns
    assert "position" in store.columns

    assert "emb_local" not in store.columns
    assert "emb_medium" not in store.columns


def test_get_returns_observation(
    tmp_path,
) -> None:
    write_test_corpus(tmp_path)

    store = ParquetObservationStore(tmp_path)

    observation = store.get(1002)

    assert observation["event_id"] == 1002
    assert observation["document_id"] == "doc-b"
    assert observation["token"] == "pale"
    assert observation["position"] == 20


def test_get_many_ordered_preserves_requested_order(
    tmp_path,
) -> None:
    write_test_corpus(tmp_path)

    store = ParquetObservationStore(tmp_path)

    observations = store.get_many_ordered(
        [1003, 1001, 1002],
    )

    assert [item["event_id"] for item in observations] == [
        1003,
        1001,
        1002,
    ]


def test_get_many_ordered_preserves_missing_ids(
    tmp_path,
) -> None:
    write_test_corpus(tmp_path)

    store = ParquetObservationStore(tmp_path)

    observations = store.get_many_ordered(
        [1003, 9999, 1001],
    )

    assert observations[0]["event_id"] == 1003
    assert observations[1] is None
    assert observations[2]["event_id"] == 1001


def test_get_missing_observation_raises_key_error(
    tmp_path,
) -> None:
    write_test_corpus(tmp_path)

    store = ParquetObservationStore(tmp_path)

    try:
        store.get(9999)
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Expected missing observation to raise KeyError"
        )

