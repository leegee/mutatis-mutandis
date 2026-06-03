/**
 * src/state/tier2data.store.ts
 *
 * Replaced implementation: loads the SQLite .db asset instead of JSON.
 *
 * The store no longer holds concept data in memory.  It only tracks
 * initialisation state.  All data access goes through selectors.ts which
 * issues SQL queries on demand.
 *
 * CORPUS_TIER2_DB_URL should point to the .db file served as a static asset,
 * e.g. "/data/tier2_concept_neighbours.db".  Add it to corpus_config.ts.
 */

import { createSignal } from "solid-js";
import { initDb } from "../services/db/dbh";
import { CORPUS_TIER2_DB_URL } from "../corpus_config";

// ---------------------------------------------------------------------------
// Public signals
// ---------------------------------------------------------------------------

/** True once the database file has been fetched and opened. */
export const [dbReady, setDbReady] = createSignal(false);

/** Non-null when initialisation fails. */
export const [dbError, setDbError] = createSignal<string | null>(null);

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

/**
 * Fetch the .db asset, write to OPFS, and open it.
 * Call once at app startup (e.g. in App.tsx before rendering).
 * Safe to call multiple times — the underlying initDb is idempotent.
 */
export async function loadTier2Data(): Promise<void> {
  try {
    await initDb(CORPUS_TIER2_DB_URL);
    setDbReady(true);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setDbError(msg);
    console.error("[tier2] failed to load database:", msg);
  }
}
