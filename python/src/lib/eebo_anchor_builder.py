# lib/eebo_anchor_builder_stable.py

"""
Build stable anchor words for temporal vector alignment in EEBO slices.

Only words that are likely **semantically stable across time** are kept:
- Function words (e.g., prepositions, pronouns, determiners, conjunctions, particles)
- High-frequency terms that appear consistently across slices
- Excludes proper nouns, rare words, and ideologically/religiously loaded nouns/verbs
This ensures Orthogonal Procrustes alignment uses only stable anchors,
allowing content words to drift naturally.
"""

import os
import json
from collections import Counter
import spacy
import lib.eebo_config as config

MIN_FREQ = 50  # minimum frequency per slice
POS_KEEP = {"NOUN", "PRON", "ADP", "DET", "CONJ", "PART"}  # stable POS for anchors
FUNC_POS = {"PRON", "ADP", "DET", "CONJ", "PART"}  # strictly function words

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # only tagger needed

def load_slice_tokens(slice_path):
    """Load a slice text file and tokenize by whitespace."""
    with open(slice_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    return tokens

def pos_filter(tokens, pos_keep=None):
    """Filter tokens by stable POS. Optionally restrict to function words."""
    pos_keep = pos_keep or POS_KEEP
    doc = nlp(" ".join(tokens))
    return [t.text for t in doc if t.pos_ in pos_keep]

def build_anchors():
    """Build stable anchors for all slices and save to ALIGNMENT_ANCHORS_FILE."""
    anchors_dict = {}
    slice_files = sorted(os.listdir(config.SLICES_DIR))

    for fname in slice_files:
        slice_id = os.path.splitext(fname)[0]
        slice_path = os.path.join(config.SLICES_DIR, fname)
        print(f"Processing slice: {slice_id}")

        tokens = load_slice_tokens(slice_path)
        freqs = Counter(tokens)

        # Keep tokens above frequency threshold
        frequent_tokens = [w for w, c in freqs.items() if c >= MIN_FREQ]

        # Restrict to function words only to ensure semantic stability
        stable_anchors = pos_filter(frequent_tokens, pos_keep=FUNC_POS)

        # Optional: weight by frequency stability within slice
        total = sum(freqs[w] for w in stable_anchors)
        weights = [freqs[w]/total for w in stable_anchors] if total > 0 else []

        anchors_dict[slice_id] = {
            "anchors": stable_anchors,
            "weights": weights,
            "pos": [t.pos_ for t in nlp(" ".join(stable_anchors))]
        }

    # Save cache
    os.makedirs(config.INDEXES_DIR, exist_ok=True)
    with open(config.ALIGNMENT_ANCHORS_FILE, "w", encoding="utf-8") as f:
        json.dump(anchors_dict, f, indent=2)

    print(f"Stable anchors saved to {config.ALIGNMENT_ANCHORS_FILE}")
    return anchors_dict

def get_anchors():
    """
    Return anchors dictionary for alignment.
    - Loads from config.ALIGNMENT_ANCHORS_FILE if it exists
    - Otherwise calls build_anchors() to generate and cache them
    """
    if os.path.exists(config.ALIGNMENT_ANCHORS_FILE):
        print(f"Loading stable anchors from {config.ALIGNMENT_ANCHORS_FILE}")
        with open(config.ALIGNMENT_ANCHORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("Stable anchors file not found. Building anchors...")
        return build_anchors()

if __name__ == "__main__":
    anchors = get_anchors()
