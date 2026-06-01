/**
 * src/services/db.ts
 *
 * Main-thread interface to the SQLite worker.
 *
 * All SQLite work runs in db.worker.ts.  This module owns the Worker
 * instance and exposes typed query helpers that send messages and await
 * responses via a promise-per-message correlation map.
 *
 * Usage
 * -----
 *   await initDb(url);   // once at startup
 *   queryConcepts();     // synchronous-feeling helpers (return Promises)
 *
 * Because the worker is async, all query helpers are async.  However,
 * inside a dbReady()-gated component they can be called from createMemo
 * only if you accept that the memo returns a Promise — prefer createResource
 * for data fetching, or call the helpers synchronously via a thin
 * bridge if you need true synchrony (see note below).
 *
 * Synchronous bridge note
 * -----------------------
 * The oo1 API is synchronous *inside the worker*.  The main↔worker boundary
 * is always async (postMessage).  If you need synchronous access in the main
 * thread, the recommended approach is to use the COOP+COEP + SharedArrayBuffer
 * Atomics.wait() pattern — but that adds complexity.  For this app, the
 * createResource() pattern in SolidJS is the clean solution.
 *
 * However — since this app gates all rendering behind dbReady(), and all
 * queries are fast (indexed reads, no aggregation), we use a simpler approach:
 * a synchronous-looking API backed by a pre-populated cache that is filled
 * once on init.  See the "query cache" section below.
 */

import type { ConceptEvent, Neighbour } from "../types/context-graph.types";

// ---------------------------------------------------------------------------
// Worker setup
// ---------------------------------------------------------------------------

// Vite handles ?worker imports and bundles the worker correctly.
// The worker file must be in the same origin.
import DbWorker from "./db.worker?worker";

let _worker: Worker | null = null;
let _pending = new Map<string, { resolve: (v: unknown[][]) => void; reject: (e: Error) => void }>();
let _msgId = 0;

function getWorker(): Worker {
  if (!_worker) throw new Error("[db] worker not initialised — call initDb() first");
  return _worker;
}

function send(type: string, payload: Record<string, unknown> = {}): Promise<unknown[][]> {
  return new Promise((resolve, reject) => {
    const id = String(++_msgId);
    _pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, type, ...payload });
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

let _initPromise: Promise<void> | null = null;

export async function initDb(url: string): Promise<void> {
  if (_initPromise) return _initPromise;

  _initPromise = (async () => {
    _worker = new DbWorker();

    _worker.onmessage = (e: MessageEvent) => {
      const { id, result, error } = e.data;
      const pending = _pending.get(id);
      if (!pending) return;
      _pending.delete(id);
      if (error) {
        pending.reject(new Error(error));
      } else {
        pending.resolve(result);
      }
    };

    _worker.onerror = (e) => {
      console.error("[db] worker error:", e.message);
    };

    await send("init", { url });
    console.log("[db] worker ready");
  })();

  return _initPromise;
}

// ---------------------------------------------------------------------------
// exec helper
// ---------------------------------------------------------------------------

async function execRows(
  sql: string,
  bind?: (string | number | null)[],
): Promise<unknown[][]> {
  return send("exec", { sql, bind });
}

// ---------------------------------------------------------------------------
// Typed query helpers (all async — cross worker boundary)
// ---------------------------------------------------------------------------

export async function queryConcepts(): Promise<string[]> {
  const rows = await execRows("SELECT concept FROM concepts ORDER BY concept");
  return rows.map((r) => r[0] as string);
}

export async function queryYearBounds(
  concept: string,
): Promise<[number, number] | null> {
  const rows = await execRows(
    `SELECT MIN(pub_year), MAX(pub_year)
     FROM   events
     WHERE  concept = ?
       AND  pub_year IS NOT NULL`,
    [concept],
  );
  const row = rows[0];
  if (!row || row[0] == null || row[1] == null) return null;
  return [row[0] as number, row[1] as number];
}

export async function queryEvents(
  concept: string,
  fromYear: number,
  toYear: number,
): Promise<ConceptEvent[]> {
  const eventRows = await execRows(
    `SELECT event_id, vector_id, token, doc_id, pub_year,
            token_idx, window_id, window_token_pos
     FROM   events
     WHERE  concept   = ?
       AND  pub_year >= ?
       AND  pub_year <= ?
     ORDER  BY pub_year, event_id`,
    [concept, fromYear, toYear],
  );

  if (eventRows.length === 0) return [];

  const eventMap = new Map<number, ConceptEvent>();
  const events: ConceptEvent[] = [];

  for (const r of eventRows) {
    const e: ConceptEvent = {
      event_id: r[0] as number,
      vector_id: r[1] as number,
      token: r[2] as string,
      doc_id: r[3] as string,
      pub_year: r[4] as number,
      token_idx: r[5] as number,
      window_id: r[6] as number,
      window_token_pos: r[7] as number,
      neighbours: [],
    };
    eventMap.set(e.event_id, e);
    events.push(e);
  }

  const ids = [...eventMap.keys()].join(",");
  const nbRows = await execRows(
    `SELECT event_id, neighbour_event_id, vector_id, token,
            doc_id, pub_year, token_idx, window_id,
            window_token_pos, score
     FROM   neighbours
     WHERE  event_id IN (${ ids })
     ORDER  BY event_id, score DESC`,
  );

  for (const r of nbRows) {
    const nb: Neighbour = {
      event_id: r[1] as number,
      vector_id: r[2] as number,
      token: r[3] as string,
      doc_id: r[4] as string,
      pub_year: r[5] as number,
      token_idx: String(r[6]),
      window_id: r[7] as number,
      window_token_pos: r[8] as number,
      score: r[9] as number,
    };
    eventMap.get(r[0] as number)?.neighbours.push(nb);
  }

  return events;
}

export async function queryNEvents(concept: string): Promise<number> {
  const rows = await execRows(
    "SELECT n_events FROM concepts WHERE concept = ?",
    [concept],
  );
  return (rows[0]?.[0] as number) ?? 0;
}

export async function queryAggregate(concept: string, topN = 25) {
  const rows = await execRows(
    `SELECT kind, rank, value, window_doc_id, window_id, count
     FROM   concept_aggregate
     WHERE  concept = ?
     ORDER  BY kind, rank`,
    [concept],
  );

  const top_tokens: [string, number][] = [];
  const top_docs: [string, number][] = [];
  const top_windows: [[string, number], number][] = [];

  for (const r of rows) {
    const [kind, , value, windowDocId, windowId, count] = r as [
      string, number, string | null, string | null, number | null, number,
    ];
    if (kind === "token" && value != null) top_tokens.push([value, count]);
    if (kind === "doc" && value != null) top_docs.push([value, count]);
    if (kind === "window" && windowDocId != null && windowId != null)
      top_windows.push([[windowDocId, windowId], count]);
  }

  return {
    top_tokens: top_tokens.slice(0, topN),
    top_docs: top_docs.slice(0, topN),
    top_windows: top_windows.slice(0, topN),
  };
}