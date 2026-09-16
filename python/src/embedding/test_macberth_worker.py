from __future__ import annotations

import numpy as np

from embedding.macberth_worker import MacBERThEventEmbedder
from lib.corpus_db import get_connection
from lib.macberth import load_macberth


CORPUS = "clmet"
DOC_ID = "CLMET31"

TARGET_POSITIONS = {
    127,
    128,
    129,
    255,
    256,
    257,
    383,
    384,
    385,
    511,
    512,
    513,
}


def main() -> None:
    conn = get_connection()

    try:
        mac = load_macberth()

        embedder = MacBERThEventEmbedder(
            conn,
            mac,
            batch_size=4,
            scale="local",
            mask_targets=False,
        )

        document = embedder._load_document(
            CORPUS,
            DOC_ID,
        )

        if document is None:
            raise RuntimeError(
                f"document not found: {CORPUS}/{DOC_ID}"
            )

        print(
            f"document: {document.corpus}/{document.doc_id}"
        )
        print(f"tokens: {len(document.rows)}")
        print(
            "target positions:",
            sorted(TARGET_POSITIONS),
        )

        if max(TARGET_POSITIONS) >= len(document.rows):
            raise RuntimeError(
                "Test target is outside document."
            )

        embedded = embedder.embed_document_targets(
            document=document,
            target_positions=TARGET_POSITIONS,
        )

        print("embedded:", len(embedded))

        if set(embedded) != TARGET_POSITIONS:
            missing = TARGET_POSITIONS - set(embedded)
            extra = set(embedded) - TARGET_POSITIONS

            raise AssertionError(
                f"embedding positions mismatch: "
                f"missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        for position in sorted(embedded):
            value = embedded[position]
            row = document.rows[position]

            print(
                f"position={position:5d} "
                f"token_idx={row.token_idx:5d} "
                f"token={row.token!r:20s} "
                f"window_id={value.window_id:5d} "
                f"window_token_pos={value.window_token_pos:3d} "
                f"shape={value.vector.shape} "
                f"dtype={value.vector.dtype}"
            )

            if value.vector.dtype != np.float32:
                raise AssertionError(
                    f"unexpected dtype: {value.vector.dtype}"
                )

            if value.vector.shape != (768,):
                raise AssertionError(
                    f"unexpected vector shape: "
                    f"{value.vector.shape}"
                )

            if not np.isfinite(value.vector).all():
                raise AssertionError(
                    f"non-finite vector at position {position}"
                )

            if value.window_id < 0:
                raise AssertionError(
                    f"invalid window_id: {value.window_id}"
                )

            if not (
                0 <= value.window_token_pos < 512
            ):
                raise AssertionError(
                    "invalid window_token_pos: "
                    f"{value.window_token_pos}"
                )

        print()
        print("PASS: MacBERTh worker boundary embedding path")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
