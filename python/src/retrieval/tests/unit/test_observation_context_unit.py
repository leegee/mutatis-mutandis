from __future__ import annotations

import pytest

from retrieval.parquet_context import (
    ContextToken,
    ObservationContext,
)


def test_observation_context_exposes_stable_core_fields() -> None:
    observation = {
        "corpus": "eebo",
        "doc_id": "A03930",
        "token": "preachers",
        "token_idx": 42,
        "pub_year": 1625,
    }

    context = ObservationContext(
        event_id=123,
        distance=0.25,
        observation=observation,
        before=("one", "two"),
        after=("four", "five"),
    )

    assert context.event_id == 123
    assert context.distance == 0.25
    assert context.observation is observation
    assert context.before == (
        "one",
        "two",
    )
    assert context.after == (
        "four",
        "five",
    )


def test_observation_context_text_places_observation_between_context() -> None:
    context = ObservationContext(
        event_id=123,
        distance=0.25,
        observation={
            "corpus": "eebo",
            "doc_id": "A03930",
            "token": "preachers",
            "token_idx": 42,
        },
        before=(
            ContextToken(
                corpus="eebo",
                doc_id="A03930",
                token_idx=40,
                token="one",
            ),
            ContextToken(
                corpus="eebo",
                doc_id="A03930",
                token_idx=41,
                token="two",
            ),
        ),
        after=(
            ContextToken(
                corpus="eebo",
                doc_id="A03930",
                token_idx=43,
                token="four",
            ),
            ContextToken(
                corpus="eebo",
                doc_id="A03930",
                token_idx=44,
                token="five",
            ),
        ),
    )

    assert context.before[0].token == "one"
    assert context.before[0].token_idx == 40

    assert context.after[0].token == "four"
    assert context.after[0].token_idx == 43

    assert context.text == "one two preachers four five"


def test_observation_context_text_handles_empty_context() -> None:
    context = ObservationContext(
        event_id=123,
        distance=0.25,
        observation={
            "corpus": "eebo",
            "doc_id": "A03930",
            "token": "preachers",
            "token_idx": 42,
        },
        before=(),
        after=(),
    )

    assert context.text == "preachers"


def test_observation_context_is_immutable() -> None:
    context = ObservationContext(
        event_id=123,
        distance=0.25,
        observation={
            "corpus": "eebo",
            "doc_id": "A03930",
            "token": "preachers",
            "token_idx": 42,
        },
        before=(),
        after=(),
    )

    with pytest.raises(AttributeError):
        context.distance = 0.5
