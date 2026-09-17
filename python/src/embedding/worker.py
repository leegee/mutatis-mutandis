from __future__ import annotations

import argparse
import time

from embedding.macberth_worker import MacBERThEventEmbedder
from embedding.work_queue import (
    begin_embedding,
    claim_next_work,
    complete_work,
    record_inventory,
)
from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.macberth import MACBERTH_MODEL_NAME, load_macberth
from tier1.tier1_corpus2events import (
    EmbeddedObservation,
    EventWriter,
    Observation,
)


def get_macberth_model_id() -> int:
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    model_id,
                    embedding_dimension
                FROM embedding.models
                WHERE model_key = %s
                """,
                (MACBERTH_MODEL_NAME,),
            )
            row = cur.fetchone()

        if row is None:
            raise RuntimeError(
                "MacBERTh is not registered in embedding.models: "
                f"{MACBERTH_MODEL_NAME!r}"
            )

        model_id, embedding_dimension = row

        if embedding_dimension != 768:
            raise RuntimeError(
                "Registered MacBERTh model has unexpected embedding "
                f"dimension: {embedding_dimension}"
            )

        return int(model_id)

    finally:
        conn.close()


def process_one(
    model_id: int,
    worker_id: str,
    mac,
) -> bool:
    conn = get_connection()

    try:
        work = claim_next_work(
            model_id=model_id,
            worker_id=worker_id,
        )

        if work is None:
            return False

        print(
            f"Claimed work {work.work_id}: "
            f"{len(work.event_ids)} events, "
            f"scale={work.scale}"
        )

        begin_embedding(
            work_id=work.work_id,
            worker_id=worker_id,
        )

        embedder = MacBERThEventEmbedder(
            conn,
            mac,
            batch_size=64,
            scale=work.scale,
        )

        vectors = embedder.embed_events(work.event_ids)

        expected_ids = set(work.event_ids)
        actual_ids = set(vectors)

        missing = expected_ids - actual_ids
        unexpected = actual_ids - expected_ids

        if missing:
            raise RuntimeError(
                f"MacBERTh returned no vector for event IDs: "
                f"{sorted(missing)[:20]}"
            )

        if unexpected:
            raise RuntimeError(
                f"MacBERTh returned unexpected event IDs: "
                f"{sorted(unexpected)[:20]}"
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.event_id,
                    e.corpus,
                    e.doc_id,
                    e.token_idx,
                    e.pub_year,
                    e.scale,
                    e.window_id,
                    e.window_token_pos
                FROM events AS e
                WHERE e.event_id = ANY(%s)
                ORDER BY e.event_id
                """,
                (list(work.event_ids),),
            )
            rows = cur.fetchall()

        if len(rows) != len(work.event_ids):
            raise RuntimeError(
                "PostgreSQL returned a different number of events "
                f"than the work item: "
                f"expected={len(work.event_ids)}, "
                f"actual={len(rows)}"
            )

        observations: list[EmbeddedObservation] = []

        for (
            event_id,
            corpus,
            doc_id,
            token_idx,
            pub_year,
            scale,
            window_id,
            window_token_pos,
        ) in rows:
            if scale != work.scale:
                raise RuntimeError(
                    f"Scale mismatch for event {event_id}: "
                    f"work={work.scale!r}, event={scale!r}"
                )

            observation = Observation(
                event_id=int(event_id),
                corpus=corpus,
                doc_id=doc_id,
                token_idx=int(token_idx),
                token="",
                pub_year=int(pub_year),
                scale=scale,
                window_id=int(window_id),
                window_token_pos=int(window_token_pos),
            )

            observations.append(
                EmbeddedObservation(
                    observation=observation,
                    vectors={
                        scale: vectors[int(event_id)],
                    },
                )
            )

        writer = EventWriter(
            conn,
            LANCE_INDEXES_DIR,
        )

        writer.write_lance(observations)

        for event_id in work.event_ids:
            record_inventory(
                work_id=work.work_id,
                worker_id=worker_id,
                event_id=event_id,
                embedding_key=(
                    f"{work.model_id}:{work.scale}:{event_id}"
                ),
            )

        complete_work(
            work_id=work.work_id,
            worker_id=worker_id,
        )

        print(
            f"Completed work {work.work_id}: "
            f"{len(work.event_ids)} vectors written"
        )

        return True

    finally:
        conn.close()


def run(
    model_id: int,
    worker_id: str,
    one: bool,
) -> None:
    mac = load_macberth()

    while True:
        processed = process_one(
            model_id=model_id,
            worker_id=worker_id,
            mac=mac,
        )

        if one or not processed:
            return

        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process PostgreSQL embedding work with MacBERTh."
    )
    parser.add_argument(
        "--worker-id",
        required=True,
    )
    parser.add_argument(
        "--one",
        action="store_true",
        help="Process at most one work item.",
    )

    args = parser.parse_args()

    macberth_model_id = get_macberth_model_id()

    run(
        model_id=macberth_model_id,
        worker_id=args.worker_id,
        one=args.one,
    )


if __name__ == "__main__":
    main()
