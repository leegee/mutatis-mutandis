#!/usr/bin/env python

import argparse
import json
import math


def field_entropy(rows):

    weights = [
        row.get("weight_norm", row.get("weight", 0))
        for row in rows
        if row.get("weight_norm", row.get("weight", 0)) > 0
    ]

    if len(weights) < 2:
        return 0.0

    total = sum(weights)

    entropy = -sum(
        (w / total) * math.log(w / total)
        for w in weights
    )

    return entropy / math.log(len(weights))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf8") as f:
        data = json.load(f)

    concept = data.get(
        "concept",
        "unknown",
    )

    periods = data.get(
        "trajectory",
        {},
    )

    print()
    print(f"CONCEPT: {concept}")
    print("=" * 80)


    for period, rows in periods.items():

        if isinstance(rows, dict):
            rows = list(rows.values())

        rows = sorted(
            rows,
            key=lambda x: x.get(
                "weight_norm",
                x.get(
                    "weight",
                    0,
                ),
            ),
            reverse=True,
        )

        entropy = field_entropy(rows)

        print()
        print(
            f"PERIOD: {period}"
            f"  entropy={entropy:.3f}"
        )
        print("-" * 80)

        for row in rows[:5]:

            print(
                f"{row['token']:<25}"
                f"{row.get('weight_norm', row.get('weight', 0)):10.4f}"
                f"  n={row.get('events', '?')}"
                f"  years={row.get('years', [])}"
            )


if __name__ == "__main__":
    main()