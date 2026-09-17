# embedding/test_queue.py
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
TEST_SCALE = "local"


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
    created = create_work(
        model_id=test_model.model_id,
        scale=TEST_SCALE,
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
    assert work.scale == TEST_SCALE
    assert len(work.event_ids) == 1

    event_id = work.event_ids[0]

    begin_embedding(
        work_id=work.work_id,
        worker_id=worker_id,
    )

    embedding_key = (
        f"{test_model.model_id}:{TEST_SCALE}:{event_id}"
    )

    inserted = record_inventory(
        work_id=work.work_id,
        worker_id=worker_id,
        event_id=event_id,
        embedding_key=embedding_key,
    )

    assert inserted is True

    # Repeating the same operation must be idempotent.
    inserted = record_inventory(
        work_id=work.work_id,
        worker_id=worker_id,
        event_id=event_id,
        embedding_key=embedding_key,
    )

    assert inserted is False

    # Another worker cannot write inventory for this work.
    with pytest.raises(RuntimeError, match="owned by"):
        record_inventory(
            work_id=work.work_id,
            worker_id="pytest-worker-b",
            event_id=event_id,
            embedding_key=embedding_key,
        )

    # Obtain a real event belonging to a different work item.
    other_work = claim_next_work(
        model_id=test_model.model_id,
        worker_id="pytest-worker-other",
    )

    assert other_work is not None
    assert other_work.scale == TEST_SCALE
    assert len(other_work.event_ids) == 1

    other_event_id = other_work.event_ids[0]

    # The other work is deliberately left unfinished. Fixture cleanup
    # removes it after the test.
    with pytest.raises(RuntimeError, match="does not belong"):
        record_inventory(
            work_id=work.work_id,
            worker_id=worker_id,
            event_id=other_event_id,
            embedding_key=(
                f"{test_model.model_id}:{TEST_SCALE}:{other_event_id}"
            ),
        )

    complete_work(
        work_id=work.work_id,
        worker_id=worker_id,
    )


def test_complete_requires_all_inventory(test_model):
    created = create_work(
        model_id=test_model.model_id,
        scale=TEST_SCALE,
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
    assert work.scale == TEST_SCALE
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
        embedding_key=(
            f"{test_model.model_id}:{TEST_SCALE}:{first_event_id}"
        ),
    )

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
        embedding_key=(
            f"{test_model.model_id}:{TEST_SCALE}:{second_event_id}"
        ),
    )

    complete_work(
        work_id=work.work_id,
        worker_id=worker_id,
    )


def test_scale_is_part_of_inventory_identity(test_model):
    """
    The same event may have one embedding for each scale.

    Therefore (event_id, model_id) is not sufficient inventory identity.
    """
    created = create_work(
        model_id=test_model.model_id,
        scale="local",
        batch_size=1,
        limit=1,
    )

    assert created == 1

    local_work = claim_next_work(
        model_id=test_model.model_id,
        worker_id="pytest-local",
    )

    assert local_work is not None
    event_id = local_work.event_ids[0]

    begin_embedding(
        work_id=local_work.work_id,
        worker_id="pytest-local",
    )

    local_key = (
        f"{test_model.model_id}:local:{event_id}"
    )

    assert record_inventory(
        work_id=local_work.work_id,
        worker_id="pytest-local",
        event_id=event_id,
        embedding_key=local_key,
    ) is True

    complete_work(
        work_id=local_work.work_id,
        worker_id="pytest-local",
    )

    # The same event can independently require a medium-scale embedding.
    # We need to create work specifically for that scale; the local
    # inventory must not satisfy it.
    created = create_work(
        model_id=test_model.model_id,
        scale="medium",
        batch_size=1,
        limit=1,
    )

    assert created == 1

    medium_work = claim_next_work(
        model_id=test_model.model_id,
        worker_id="pytest-medium",
    )

    assert medium_work is not None
    assert medium_work.scale == "medium"
    assert medium_work.event_ids == (event_id,)

    begin_embedding(
        work_id=medium_work.work_id,
        worker_id="pytest-medium",
    )

    medium_key = (
        f"{test_model.model_id}:medium:{event_id}"
    )

    assert record_inventory(
        work_id=medium_work.work_id,
        worker_id="pytest-medium",
        event_id=event_id,
        embedding_key=medium_key,
    ) is True

    complete_work(
        work_id=medium_work.work_id,
        worker_id="pytest-medium",
    )