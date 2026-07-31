#!/usr/bin/env python

"""
test_eebo_faiss_load.py

Smoke test for loading persisted EEBO FAISS indices.
"""

from lib.eebo_faiss import EeboFaissIndex


def main():


    # Load entire available corpus
    index = EeboFaissIndex.load_all()

    logger.debug(f"Loaded years: {sorted(index.keys())}")

    for year, scales in index.items():
        assert set(scales) == {"local", "medium", "broad"}

        for scale, idx in scales.items():
            assert isinstance(idx, EeboFaissIndex)
            assert idx.ntotal > 0
            print( f"{year} {scale}: {idx.ntotal:,} vectors dim={idx.dim}" )


    # Load selected years only
    years = sorted(index.keys())[:2]
    subset = EeboFaissIndex.load_range( years=years, )

    print("\nSelected years:")
    for year, scales in subset.items():
        print(
            year,
            {
                scale: idx.ntotal
                for scale, idx in scales.items()
            }
        )


    # Basic retrieval sanity check
    year = years[0]
    local = subset[year]["local"]

    print("\nLocal index:")
    print(f"year={year}")
    print(f"vectors={local.ntotal}")
    print(f"dim={local.dim}")


if __name__ == "__main__":
    main()
