# embedding/work_queue.py
#
# pytest src/embedding/test_queue.py -v

from __future__ import annotations

import uuid

import pytest

from embedding.work_queue import (
    begin_embedding,
    claim_next_work,
    complete_work,
    create_work,
    record_inventory,
    register_model,
    reset_model_work,
)


TEST_DIMENSION = 768


@pytest.fixture
def test_model():
    revision = f"pytest-{uuid.uuid4()}"

    model = register_model(
        model_key="emanjavacas/MacBERTh",
        model_revision=revision,
        embedding_dimension=TEST_DIMENSION,
    )

    reset_model_work(
        model_id=model.model_id,
        clear_inventory=True,
    )

    yield model

    reset_model_work(
        model_id=model.model_id,
        clear_inventory=True,
    )


def test_register_model_rejects_dimension_mismatch():
    revision = f"pytest-dimension-{uuid.uuid4()}"

    model = register_model(
        model_key="emanjavacas/MacBERTh",
        model_revision=revision,
        embedding_dimension=768,
    )

    assert model.embedding_dimension == 768

    with pytest.raises(ValueError, match="embedding dimension mismatch"):
        register_model(
            model_key="emanjavacas/MacBERTh",
            model_revision=revision,
            embedding_dimension=1024,
        )

    reset_model_work(
        model_id=model.model_id,
        clear_inventory=True,
    )


def test_inventory_requires_owned_embedding_work(test_model):
    # Create two one-event work items. This gives us:
    #   - one event belonging to the claimed work
    #   - another real event that does not belong to it
    created = create_work(
        model_id=test_model.model_id,
        batch_size=1,
        limit=2,
    )

    assert created == 2

    worker_id = "pytest-worker-a"

    work = claim_next_work(
        model_id=test_model.model_id,
        worker_id=worker_id,
    )

    assert work is not None
    assert len(work.event_ids) == 1

    event_id = work.event_ids[0]

    begin_embedding(
        work_id=work.work_id,
        worker_id=worker_id,
    )

    # A valid inventory record must succeed.
    inserted = record_inventory(
        work_id=work.work_id,
        worker_id=worker_id,
        event_id=event_id,
        embedding_key=f"{test_model.model_id}:{event_id}",
    )

    assert inserted is True

    # Repeating the same operation must be idempotent.
    inserted = record_inventory(
        work_id=work.work_id,
        worker_id=worker_id,
        event_id=event_id,
        embedding_key=f"{test_model.model_id}:{event_id}",
    )

    assert inserted is False

    # Another worker cannot write inventory for this work.
    with pytest.raises(RuntimeError, match="owned by"):
        record_inventory(
            work_id=work.work_id,
            worker_id="pytest-worker-b",
            event_id=event_id,
            embedding_key=f"{test_model.model_id}:{event_id}",
        )

    # Obtain a real event belonging to a different work item.
    other_work = claim_next_work(
        model_id=test_model.model_id,
        worker_id="pytest-worker-other",
    )

    assert other_work is not None
    assert len(other_work.event_ids) == 1

    other_event_id = other_work.event_ids[0]

    # The other work is deliberately left unfinished. Fixture cleanup
    # removes it after the test.
    with pytest.raises(RuntimeError, match="does not belong"):
        record_inventory(
            work_id=work.work_id,
            worker_id=worker_id,
            event_id=other_event_id,
            embedding_key=f"{test_model.model_id}:{other_event_id}",
        )

    # The first work contains one event, and that event is inventoried,
    # so completion should now succeed.
    complete_work(
        work_id=work.work_id,
        worker_id=worker_id,
    )


def test_complete_requires_all_inventory(test_model):
    created = create_work(
        model_id=test_model.model_id,
        batch_size=2,
        limit=2,
    )

    assert created == 1

    worker_id = "pytest-worker"

    work = claim_next_work(
        model_id=test_model.model_id,
        worker_id=worker_id,
    )

    assert work is not None
    assert len(work.event_ids) == 2

    begin_embedding(
        work_id=work.work_id,
        worker_id=worker_id,
    )

    first_event_id, second_event_id = work.event_ids

    record_inventory(
        work_id=work.work_id,
        worker_id=worker_id,
        event_id=first_event_id,
        embedding_key=f"{test_model.model_id}:{first_event_id}",
    )

    # One of two observations is inventoried, so completion must fail.
    with pytest.raises(RuntimeError, match="incomplete: 1/2 embeddings"):
        complete_work(
            work_id=work.work_id,
            worker_id=worker_id,
        )

    # The failed completion must not destroy the embedding state.
    record_inventory(
        work_id=work.work_id,
        worker_id=worker_id,
        event_id=second_event_id,
        embedding_key=f"{test_model.model_id}:{second_event_id}",
    )

    complete_work(
        work_id=work.work_id,
        worker_id=worker_id,
    )

