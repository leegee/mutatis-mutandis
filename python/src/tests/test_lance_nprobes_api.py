# test_lance_nprobes_api

from __future__ import annotations

import inspect

import numpy as np

from lib.corpus_config import EVENTSTORE_T1_PATH, LANCE_INDEXES_DIR
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.models import SearchSpace
from tier1.observation_store_api import SCALES, open_observation_lookup


SEED_EVENT_ID = 8837468744104859267


def main() -> None:
    print("Opening Tier 1 lookup...")

    lookup = open_observation_lookup(EVENTSTORE_T1_PATH)

    print("Opening Lance index...")

    store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR,
        available_years=lookup.available_years,
        available_scales=SCALES,
        dimensions=768,
        nprobes=20,
        model="macberth",
    )

    indexes = store.get(
        SearchSpace(
            years=None,
            scale=("local",),
        )
    )

    index = indexes["local"]

    print()
    print("LanceObservationIndex")
    print("---------------------")
    print(f"type: {type(index)}")
    print(f"_nprobes: {index._nprobes}")

    vector = index.reconstruct(SEED_EVENT_ID)

    print()
    print("Query builder")
    print("-------------")

    request = index._table.search(
        np.asarray(vector, dtype=np.float32),
        vector_column_name="vector",
    )

    print(f"type: {type(request)}")

    print()
    print("Probe-related attributes/methods")
    print("--------------------------------")

    names = sorted(
        name
        for name in dir(request)
        if "probe" in name.lower()
    )

    if names:
        for name in names:
            print(f"  {name}")
    else:
        print("  NONE")

    print()
    print("Search-related attributes/methods")
    print("----------------------------------")

    names = sorted(
        name
        for name in dir(request)
        if any(
            term in name.lower()
            for term in (
                "search",
                "ivf",
                "index",
                "param",
            )
        )
    )

    for name in names:
        print(f"  {name}")

    print()
    print("Relevant method signatures")
    print("--------------------------")

    for name in sorted(dir(request)):
        if any(
            term in name.lower()
            for term in (
                "probe",
                "search",
                "param",
                "index",
            )
        ):
            attribute = getattr(request, name)

            if callable(attribute):
                try:
                    signature = inspect.signature(attribute)
                except (TypeError, ValueError):
                    signature = "<signature unavailable>"

                print(f"\n{name}{signature}")

                try:
                    doc = inspect.getdoc(attribute)
                except Exception:
                    doc = None

                if doc:
                    print(doc[:1000])

    print()
    print("Current query builder repr")
    print("--------------------------")
    print(request)

    print()
    print("Lance version")
    print("-------------")

    try:
        import lancedb

        print(
            f"lancedb.__version__: "
            f"{getattr(lancedb, '__version__', '<not exposed>')}"
        )
    except Exception as exc:
        print(f"Could not inspect lancedb version: {exc}")

    try:
        import lance

        print(
            f"lance.__version__: "
            f"{getattr(lance, '__version__', '<not exposed>')}"
        )
    except Exception as exc:
        print(f"Could not inspect lance version: {exc}")


if __name__ == "__main__":
    main()
