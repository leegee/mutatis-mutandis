#!/usr/bin/env python
import argparse
import zarr

from lib.corpus_config import ZARR_PATH, MASKED_ZARR_PATH
from lib.zarr_store_dirs import store_dirs


def check(root):
    print(f"Checking: {root}")
    any_mismatch = False

    for store_dir in store_dirs(root):
        g = zarr.open_group(str(store_dir), mode="r")
        if "events" not in g:
            print(f"  [skip] {store_dir} — no 'events' group")
            continue

        e = g["events"]
        lengths = {name: e[name].shape[0] for name in e.array_keys()}

        if len(set(lengths.values())) > 1:
            any_mismatch = True
            print(f"  [MISMATCH] {store_dir}")
            for name, n in sorted(lengths.items()):
                print(f"      {name:<20} {n:,}")
        else:
            n = next(iter(lengths.values()), 0)
            print(f"  [ok] {store_dir} — {n:,} rows, {len(lengths)} fields")

    if not any_mismatch:
        print("All stores internally consistent.")
    return any_mismatch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", action="store_true", help="Check MASKED_ZARR_PATH instead of ZARR_PATH")
    p.add_argument("--path", type=str, default=None, help="Override with an explicit path (e.g. a shard dir)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.path:
        root = args.path
    else:
        root = MASKED_ZARR_PATH if args.mask else ZARR_PATH

    found_mismatch = check(root)
    raise SystemExit(1 if found_mismatch else 0)


if __name__ == "__main__":
    main()
