#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
import numpy as np
from mb_test import OUT_PATH

DATA_FILE = OUT_PATH

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    raise ValueError("Expected JSON to be a dict keyed by token")

# Iterate over tokens in the dataset
for token, token_data in data.items():
    slices_data = token_data.get("slices")
    transitions = token_data.get("phase_transitions", {})

    if not slices_data:
        print(f"No slices for {token}, skipping")
        continue

    # Extract main time series
    years = [s["year"] for s in slices_data]
    drift = [s["drift"] for s in slices_data]
    jsd = [s["js_divergence"] for s in slices_data]
    births = [s["births"] for s in slices_data]

    plt.figure(figsize=(14, 6))
    plt.plot(years, drift, label="Drift", marker="o")
    plt.plot(years, jsd, label="JSD", marker="x")
    plt.plot(years, births, label="Births", marker="^")

    # Plot major transitions
    for t in transitions.get("major", []):
        plt.axvline(x=t["year"], color="red", linestyle="--", alpha=0.7)
        plt.text(
            t["year"], max(max(drift), max(jsd)) * 1.05,
            f"MAJOR {t['year']}", rotation=90, color="red", verticalalignment="bottom"
        )

    # Plot minor transitions
    for t in transitions.get("minor", []):
        plt.axvline(x=t["year"], color="orange", linestyle=":", alpha=0.7)
        plt.text(
            t["year"], max(max(drift), max(jsd)) * 1.05,
            f"MINOR {t['year']}", rotation=90, color="orange", verticalalignment="bottom"
        )

    # Plot single-doc spikes
    for t in transitions.get("single_doc_spikes", []):
        plt.axvline(x=t["year"], color="green", linestyle="-.", alpha=0.7)
        plt.text(
            t["year"], max(max(drift), max(jsd)) * 1.05,
            f"SINGLE {t['year']}", rotation=90, color="green", verticalalignment="bottom"
        )

    plt.title(f"Semantic drift and phase transitions for '{token}'")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
