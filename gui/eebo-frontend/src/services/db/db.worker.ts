import sqlite3InitModule from "@sqlite.org/sqlite-wasm";
import type { Database } from "@sqlite.org/sqlite-wasm";

type LogLevel = "debug" | "info" | "warn" | "error";

interface TraceEntry {
  traceId: string;
  spanId: string;
  level: LogLevel;
  event: string;
  durationMs?: number;
  data?: Record<string, unknown>;
  error?: { message: string; stack?: string; cause?: unknown };
}

console.log("[db.worker] Worker script initialized (new instance)");

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function log(entry: Omit<TraceEntry, "spanId"> & { spanId?: string }): void {
  const record: TraceEntry = { spanId: uid(), ...entry };
  const fn = record.level === "error" ? console.error
    : record.level === "warn" ? console.warn
      : record.level === "debug" ? console.debug
        : console.info;
  fn(`[db.worker][${ record.level.toUpperCase() }] ${ record.event }`, record);
}

function startSpan(traceId: string, event: string) {
  const spanId = uid();
  const t0 = performance.now();
  return {
    spanId,
    end(level: LogLevel = "debug", data?: Record<string, unknown>) {
      log({ traceId, spanId, level, event, durationMs: +(performance.now() - t0).toFixed(2), data });
    },
    fail(err: unknown, data?: Record<string, unknown>) {
      log({
        traceId,
        spanId,
        level: "error",
        event,
        durationMs: +(performance.now() - t0).toFixed(2),
        data,
        error: toErrorShape(err),
      });
    },
  };
}

function toErrorShape(err: unknown) {
  if (err instanceof Error) {
    console.groupCollapsed(`  ↳ stack`);
    console.trace(err.stack);
    console.groupEnd();
    return { message: err.message, stack: err.stack, cause: err.cause };
  }
  return { message: String(err) };
}


let DBH: Database | null = null;

// The core routine

function execRows(
  sql: string,
  bind?: (string | number | null)[],
  traceId = "untracked",
): unknown[][] {
  if (!DBH) throw new Error("database not initialised");

  const span = startSpan(traceId, "exec_rows");
  const rows: unknown[][] = [];
  try {
    DBH.exec({
      sql,
      bind,
      rowMode: "array",
      callback: (row: unknown[]) => { rows.push([...row]); },
    });
    // span.end("debug", { rowCount: rows.length, sql: truncate(sql, 120) });
    return rows;
  } catch (err) {
    span.fail(err, { sql, bind });
    throw err;
  }
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max) + "…";
}

// Init

async function init(url: string, traceId: string): Promise<void> {
  const rootSpan = startSpan(traceId, "init");

  //Bootstrap sqlite-wasm
  let sqlite3: unknown;
  {
    const span = startSpan(traceId, "sqlite_wasm_init");
    try {
      sqlite3 = await (sqlite3InitModule as any)({
        wasmMemory: new WebAssembly.Memory({ initial: 4096, maximum: 16384, shared: false }),
      });
      span.end("debug");
    } catch (err) {
      span.fail(err);
      throw err;
    }
  }

  // Fetch database file
  let buf: ArrayBuffer;
  {
    const span = startSpan(traceId, "fetch_db");
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${ res.status } ${ res.statusText } — ${ url }`);
      buf = await res.arrayBuffer();
      span.end("debug", { url, bytes: buf.byteLength });
    } catch (err) {
      span.fail(err, { url });
      throw err;
    }
  }

  // Import into OPFS and open
  {
    const span = startSpan(traceId, "opfs_import_open");
    try {
      const OpfsWlDb = (sqlite3 as any).oo1.OpfsWlDb;
      const filename = "/" + url.split("/").pop()!;
      await OpfsWlDb.importDb(filename, new Uint8Array(buf));
      DBH = new OpfsWlDb(filename, "r");
      span.end("debug", { filename });
    } catch (err) {
      span.fail(err, { url });
      throw err;
    }
  }

  // Sanity-check schema
  {
    const span = startSpan(traceId, "schema_check");
    try {
      const rows = execRows(
        `SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`,
        undefined,
        traceId,
      );
      const tables = rows.map(r => r[0] as string);
      if (!tables.includes("events")) {
        span.fail(
          new Error(`'events' table missing — found: [${ tables.join(", ") }]`),
          { tables },
        );
        throw new Error(`[db.worker] 'events' table missing. Found: [${ tables.join(", ") }]`);
      }
      span.end("debug", { tables });
    } catch (err) {
      if (!(err instanceof Error && err.message.startsWith("[db.worker]"))) {
        span.fail(err);
      }
      throw err;
    }
  }

  rootSpan.end("debug", { url });
}

// Message handler

self.onmessage = async (e: MessageEvent) => {
  const { id, type } = e.data;
  const traceId: string = (e.data.traceId as string | undefined) ?? uid();

  const span = startSpan(traceId, `msg_${ type }`);

  try {
    if (type === "init") {
      await init(e.data.url as string, traceId);
      span.end("debug");
      self.postMessage({ id, traceId, result: [] });
    }

    else if (type === "exec") {
      const rows = execRows(e.data.sql as string, e.data.bind, traceId);
      span.end("debug", { rowCount: rows.length });
      self.postMessage({ id, traceId, result: rows });
    }

    else if (type === 'prewarm') {
      const span = startSpan(traceId, "prewarm");
      try {
        const start = performance.now();

        await Promise.all([
          execRows("SELECT COUNT(*) FROM events", undefined, traceId),

          execRows(
            "SELECT nx, ny, gnx, gny FROM events LIMIT 1000",
            undefined,
            traceId
          ),

          execRows("SELECT COUNT(*) FROM neighbours", undefined, traceId),

          execRows(`
            SELECT
                e.nx,
                e.ny,
                e.gnx,
                e.gny
            FROM neighbours n
            JOIN events e
                ON e.event_id = n.neighbour_event_id
            LIMIT 2500
        `,
            undefined,
            traceId
          ),
        ]);

        const duration = performance.now() - start;
        console.log(`[db.worker] Pre-warm completed in ${ duration.toFixed(1) }ms`);

        span.end("debug", { duration });

        self.postMessage({
          id,
          traceId,
          result: { success: true, duration }
        });
      }
      catch (err) {
        span.fail(err);
        self.postMessage({
          id,
          traceId,
          error: (err as Error).message
        });
      }
    }

    else {
      const err = new Error(`unknown message type: ${ type }`);
      span.fail(err);
      throw err;
    }
  }

  catch (err) {
    const shape = toErrorShape(err);
    // span.fail already called inside init/execRows — only log here for
    // errors that escaped without being traced (e.g. unknown type above).
    self.postMessage({ id, traceId, error: shape.message });
  }
};
