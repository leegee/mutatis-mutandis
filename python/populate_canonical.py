#!/usr/bin/env python
"""
Populate canonical forms for tokens in the EEBO database using fastText and a spelling map.

- Loads fastText model(s) from configured directory
- Loads spelling_map from Postgres (manual overrides)
- Iterates over tokens in batches
- Populates tokens.canonical
"""

from pathlib import Path
import fasttext
import psycopg
from tqdm import tqdm

import lib.eebo_config as config
import lib.eebo_db


def load_spelling_map(conn):
    """Load spelling_map table into a Python dict."""
    spelling_map = {}
    with conn.cursor() as cur:
        cur.execute("SELECT variant, canonical FROM spelling_map WHERE concept_type = 'orthographic'")
        for variant, canonical in cur.fetchall():
            spelling_map[variant.lower()] = canonical.lower()
    return spelling_map

def nearest_word(word, vocab, model):
    """Return closest known word in vocab via fastText embeddings."""
    if word in vocab:
        return word
    neighbors = model.get_nearest_neighbors(word, k=5)
    for cosine, neighbor in neighbors:
        if neighbor in vocab:
            return neighbor
    return word  # fallback

def populate_canonical(conn, ft_model_path: Path):
    print(f"[INFO] Loading fastText model from {ft_model_path}")
    ft_model = fasttext.load_model(str(ft_model_path))
    vocab = set(ft_model.get_words())

    spelling_map = load_spelling_map(conn)
    print(f"[INFO] Loaded {len(spelling_map)} spelling map entries")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM tokens")
            total_tokens = cur.fetchone()[0]
            print(f"[INFO] Total tokens to process: {total_tokens}")

            offset = 0
            while offset < total_tokens:
                cur.execute(
                    "SELECT doc_id, token_idx, token FROM tokens ORDER BY doc_id, token_idx LIMIT %s OFFSET %s",
                    (config.CANONICALISATION_BATCH_SIZE, offset)
                )
                batch = cur.fetchall()
                if not batch:
                    break

                updates = []
                for doc_id, token_idx, token in batch:
                    token_lower = token.lower()
                    canonical = spelling_map.get(token_lower)
                    if not canonical:
                        canonical = nearest_word(token_lower, vocab, ft_model)
                    updates.append((canonical, doc_id, token_idx))

                # Bulk update
                args_str = ",".join(
                    cur.mogrify("(%s,%s,%s)", (can, doc_id, idx)).decode("utf-8")
                    for can, doc_id, idx in updates
                )
                update_sql = f"""
                    UPDATE tokens AS t
                    SET canonical = u.canonical
                    FROM (VALUES {args_str}) AS u(canonical, doc_id, token_idx)
                    WHERE t.doc_id = u.doc_id AND t.token_idx = u.token_idx
                """
                cur.execute(update_sql)

                print(f"[INFO] Processed tokens {offset} -> {offset + len(batch)}")
                offset += len(batch)

    print("[DONE] Canonicalization complete")

def main():
    # Pick a fastText model - for now, first .bin in MODELS_DIR
    model_files = sorted(config.MODELS_DIR.glob("*.bin"))
    if not model_files:
        print("[ERROR] No fastText model found in", config.MODELS_DIR)
        return
    ft_model_path = model_files[0]

    with eebo_db.get_connection() as conn:
        populate_canonical(conn, ft_model_path)

if __name__ == "__main__":
    main()
