import DbWorker from "./db.worker?worker";

let _worker: Worker | null = null;
let _pending = new Map<string,
  { resolve: (v: unknown[][]) => void; reject: (e: Error) => void }
>();
let _msgId = 0;

function getWorker(): Worker {
  if (!_worker)
    throw new Error("[db] worker not initialised! Call initDb() first");
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

let _initPromise: Promise<void> | null = null;

export async function initDb(url: string): Promise<void> {
  if (_initPromise) return _initPromise;
  console.log("[dbh.initDb]", url);

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
