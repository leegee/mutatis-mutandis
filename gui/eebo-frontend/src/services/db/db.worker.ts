import sqlite3InitModule from "@sqlite.org/sqlite-wasm";
import type { Database } from "@sqlite.org/sqlite-wasm";

let DBH: Database | null = null;

function execRows(sql: string, bind?: (string | number | null)[]): unknown[][] {
  if (!DBH) throw new Error("[db.worker] database not initialised");
  const rows: unknown[][] = [];
  DBH.exec({
    sql,
    bind,
    rowMode: "array",
    callback: (row: unknown[]) => { rows.push([...row]); },
  });
  return rows;
}


async function init(url: string): Promise<void> {
  const sqlite3 = await (sqlite3InitModule as any)({
    wasmMemory: new WebAssembly.Memory({ initial: 4096, maximum: 16384, shared: false }),
  });

  const buf = await fetch(url).then(r => r.arrayBuffer());
  const OpfsWlDb = (sqlite3.oo1 as any).OpfsWlDb;
  const filename = "/" + url.split("/").pop()!;
  await OpfsWlDb.importDb(filename, new Uint8Array(buf));
  DBH = new OpfsWlDb(filename, "r");

  console.debug('[db.worker] DBH', DBH)

  // Sanity check
  const rows = execRows(`SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`,);
  const tables = rows.map(_ => _[0]);

  if (!tables.includes("events")) {
    console.debug("[db.worker] tables:", tables);
    throw new Error(`[db.worker] 'events' table missing. Found: [${ tables.join(", ") }]`);
  }
}

self.onmessage = async (e: MessageEvent) => {
  const { id, type } = e.data;
  try {
    if (type === "init") {
      await init(e.data.url as string);
      self.postMessage({ id, result: [] });
    }

    else if (type === "exec") {
      const rows = execRows(e.data.sql as string, e.data.bind);
      self.postMessage({ id, result: rows });
    }

    else {
      throw new Error(`[db.worker] unknown message type: ${ type }`);
    }
  }

  catch (err) {
    if (err instanceof Error) console.error(err);
    self.postMessage({ id, error: err instanceof Error ? err.message : String(err), });
  }
};