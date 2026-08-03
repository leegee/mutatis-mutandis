import { createResource } from "solid-js";

export type WindowQueryOld = {
  corpus: string;
  docId: string;
  tokenIdx: number;
};

export type WindowQueryEventId = {
  eventId: number;
};

export type WindowQuery = WindowQueryEventId | WindowQueryOld;

export type TextWindowItem = {
  corpus: string;
  docId: string;
  tokenIdx: number;
  content: string;
}

export type WindowBatchResponse = {
  results: TextWindowItem[]
}

export function createTokenWindowBatchResource(
  event: () => WindowQuery[] | null
) {
  return createResource(event, async (queries) => {
    if (!queries || queries.length === 0) return null;
    return fetchWindowBatch(queries);
  });
}

export async function fetchWindowBatch(
  queries: WindowQuery[]
) {
  const body = JSON.stringify({ queries });

  console.trace("[tokenWindowBatchApi]", body);

  const res = await fetch(`/api/window/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });

  if (!res.ok) throw new Error("failed to fetch window batch");
  const rv = await res.json();
  // console.debug("[tokenWindowBatchApi] rv", rv);
  return rv as WindowBatchResponse;
}

