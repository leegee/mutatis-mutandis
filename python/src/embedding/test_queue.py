from embedding.work_queue import (
    begin_embedding,
    claim_next_work,
    complete_work,
    create_work,
    record_inventory,
    register_model,
    reset_model_work,
    work_summary,
)


def main() -> None:
    model = register_model(
        model_key="emanjavacas/MacBERTh",
        model_revision="v1-test",
        embedding_dimension=768,
    )

    print("model:", model)

    reset_model_work(
        model_id=model.model_id,
        clear_inventory=True,
    )

    created = create_work(
        model_id=model.model_id,
        batch_size=2,
        limit=100,
    )

    print("work items created:", created)

    print("before claim:")
    for row in work_summary(model_id=model.model_id):
        print(row)

    work = claim_next_work(
        worker_id="test-worker",
    )

    if work is None:
        raise RuntimeError("no work was available")

    print("claimed:")
    print(work)

    print("event IDs:")
    print(work.event_ids)

    begin_embedding(
        work_id=work.work_id,
        worker_id="test-worker",
    )

    print("embedding started")

    for event_id in work.event_ids:
        record_inventory(
            event_id=event_id,
            model_id=work.model_id,
            embedding_key=f"test:{work.model_id}:{event_id}",
        )

    complete_work(
        work_id=work.work_id,
        worker_id="test-worker",
    )


if __name__ == "__main__":
    main()