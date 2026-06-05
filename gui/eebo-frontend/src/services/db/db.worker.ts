/**
 * src/services/db.worker.ts
 *
 * Runs in a dedicated Worker.  Owns the sqlite-wasm instance and the open
 * database handle.  Receives query messages from the main thread and posts
 * results back.
 *
 * Message protocol
 * ----------------
 * Incoming:
 *   { id: string, type: "init",  url: string }
 *   { id: string, type: "exec",  sql: string, bind?: (string|number|null)[] }
 *
 * Outgoing:
 *   { id: string, result: unknown[][] }   — success
 *   { id: string, error:  string }        — failure
 */

import sqlite3InitModule from "@sqlite.org/sqlite-wasm";
import type { Database, Sqlite3Static } from "@sqlite.org/sqlite-wasm";

let db: Database | null = null;

async function init(url: string): Promise<void> {
  const sqlite3: Sqlite3Static = await sqlite3InitModule();
  console.log(
    "[db.worker] sqlite-wasm loaded, version",
    sqlite3.version.libVersion,
  );

  const res = await fetch(url);
  if (!res.ok)
    throw new Error(`[db.worker] fetch failed: ${ res.status } ${ url }`);
  const buf = await res.arrayBuffer();
  console.log(`[db.worker] fetched ${ buf.byteLength } bytes`);

  const hasOPFS =
    typeof navigator !== "undefined" &&
    "storage" in navigator &&
    typeof navigator.storage.getDirectory === "function";

  if (hasOPFS) {
    // Write bytes into OPFS then open as a persistent file.
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle("tier2.db", { create: true });
    // @ts-ignore — createWritable available in all OPFS-capable browsers
    const writer = await handle.createWritable();
    await writer.write(buf);
    await writer.close();

    db = new sqlite3.oo1.OpfsDb("/tier2.db", "r");
    // db.run("PRAGMA journal_mode=OFF");
    // db.run("PRAGMA synchronous=OFF");
    // db.run("PRAGMA cache_size=-65536"); // 64 MB page cache
    // db.run("PRAGMA temp_store=MEMORY");
    console.log("[db.worker] opened OPFS database");
  } else {
    throw new Error("[db.worker] OPFS unavailable");
  }

  // Sanity check
  const tables: string[] = [];
  db.exec({
    sql: `SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`,
    rowMode: "array",
    callback: (row: unknown[]) => {
      tables.push(row[0] as string);
    },
  });
  console.log("[db.worker] tables:", tables);
}

function old_execRows(sql: string, bind?: (string | number | null)[]): unknown[][] {
  if (!db) throw new Error("[db.worker] database not initialised");
  const rows: unknown[][] = [];
  db.exec({
    sql,
    bind,
    rowMode: "array",
    callback: (row: unknown[]) => {
      rows.push([...row]);
    },
  });
  return rows;
}


function execRows(sql: string, bind?: any[]): unknown[][] {
  if (!db) throw new Error("not init");

  const res = db.exec({
    sql,
    bind,
    rowMode: "array",
    returnValue: "resultRows",
  });

  return res as unknown[][];
}


self.onmessage = async (e: MessageEvent) => {
  const { id, type } = e.data;

  try {
    if (type === "init") {
      await init(e.data.url as string);
      self.postMessage({ id, result: [] });
    } else if (type === "exec") {
      const rows = execRows(e.data.sql as string, e.data.bind);
      self.postMessage({ id, result: rows });
    } else {
      throw new Error(`[db.worker] unknown message type: ${ type }`);
    }
  } catch (err) {
    self.postMessage({
      id,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
