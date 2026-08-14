# retrieval/unit/test_observation_context_unit.py

"""
pytest src/retrieval/tests/unit/test_observation_context_unit.py -v -s
"""

from __future__ import annotations

import pytest

from retrieval.parquet_context import (
    ContextToken,
    ObservationContext,
)


def test_observation_context_exposes_stable_core_fields() -> None:
    observation = {
        "event_id": 123,
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
            "event_id": 123,
            "doc_id": "A03930",
            "token": "preachers",
            "token_idx": 42,
        },
        before=(
            ContextToken(
                event_id=121,
                token="one",
                token_idx=40,
            ),
            ContextToken(
                event_id=122,
                token="two",
                token_idx=41,
            ),
        ),
        after=(
            ContextToken(
                event_id=124,
                token="four",
                token_idx=43,
            ),
            ContextToken(
                event_id=125,
                token="five",
                token_idx=44,
            ),
        ),
    )

    assert context.before[0].event_id == 121
    assert context.before[0].token == "one"
    assert context.before[0].token_idx == 40

    assert context.after[0].event_id == 124
    assert context.after[0].token == "four"
    assert context.after[0].token_idx == 43

    assert context.text == "one two preachers four five"



def test_observation_context_text_handles_empty_context() -> None:
    context = ObservationContext(
        event_id=123,
        distance=0.25,
        observation={
            "event_id": 123,
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
            "event_id": 123,
            "doc_id": "A03930",
            "token": "preachers",
            "token_idx": 42,
        },
        before=(),
        after=(),
    )

    with pytest.raises(AttributeError):
        context.event_id = 456
