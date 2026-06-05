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
 *   listConcepts();     // synchronous-feeling helpers (return Promises)
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


// WORKER SETUP
// Vite handles ?worker imports and bundles the worker correctly.
// The worker file must be in the same origin.
import DbWorker from "./db.worker?worker";

let _worker: Worker | null = null;
let _pending = new Map<
  string,
  { resolve: (v: unknown[][]) => void; reject: (e: Error) => void }
>();
let _msgId = 0;

function getWorker(): Worker {
  if (!_worker)
    throw new Error("[db] worker not initialised — call initDb() first");
  return _worker;
}

function send(
  type: string,
  payload: Record<string, unknown> = {},
): Promise<unknown[][]> {
  return new Promise((resolve, reject) => {
    const id = String(++_msgId);
    _pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, type, ...payload });
  });
}

// INIT
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

export async function execRows(
  sql: string,
  bind?: (string | number | null)[],
): Promise<unknown[][]> {
  return send("exec", { sql, bind });
}

