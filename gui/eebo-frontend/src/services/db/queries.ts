import type { Event, ConceptEvent, Neighbour } from "../../types";
import { execRows } from "./dbh";

const SQLITE_MAX_VARIABLES = 900; // stay safely under SQLite's ~999 limit

// Typed query helpers
export async function listConcepts(): Promise<string[]> {
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


export async function queryEventById(id: string): Promise<Event | null> {
  console.log("[query] queryEventById", id);
  if (typeof id !== 'string') console.trace('queryEventById received', typeof id)

  const eventRows = await execRows(
    `SELECT * FROM events WHERE event_id = ?`,
    [id],
  );

  if (eventRows.length === 0) return null;
  const row = eventRows[0];

  return {
    event_id: String(row[0]),
    concept: row[1] as string,
    vector_id: row[2] != null ? String(row[2]) : null,
    token: (row[3] as string) ?? null,
    doc_id: (row[4] as string) ?? null,
    pub_year: row[5] != null ? Number(row[5]) : null,
    token_idx: row[6] != null ? Number(row[6]) : null,
    window_id: row[7] != null ? Number(row[7]) : null,
    window_token_pos: row[8] != null ? Number(row[8]) : null,
  } as Event;
}


export async function queryEventsByIds(
  ids: string[],
): Promise<Map<string, Event>> {
  const result = new Map<string, Event>();

  // de-duplicate while preserving nothing in particular — order doesn't
  // matter since we return a Map
  const uniqueIds = Array.from(new Set(ids));
  if (uniqueIds.length === 0) return result;

  console.log("[query] queryEventsByIds", uniqueIds.length);

  for (let i = 0; i < uniqueIds.length; i += SQLITE_MAX_VARIABLES) {
    const chunk = uniqueIds.slice(i, i + SQLITE_MAX_VARIABLES);
    const placeholders = chunk.map(() => "?").join(",");

    const rows = await execRows(
      `SELECT * FROM events WHERE event_id IN (${ placeholders })`,
      chunk,
    );

    for (const row of rows) {
      const event: Event = {
        event_id: String(row[0]),
        concept: row[1] as string,
        vector_id: row[2] != null ? String(row[2]) : null,
        token: (row[3] as string) ?? null,
        doc_id: (row[4] as string) ?? null,
        pub_year: row[5] != null ? Number(row[5]) : null,
        token_idx: row[6] != null ? Number(row[6]) : null,
        window_id: row[7] != null ? Number(row[7]) : null,
        window_token_pos: row[8] != null ? Number(row[8]) : null,
      } as Event;

      result.set(event.event_id, event);
    }
  }

  return result;
}

export async function getEventsByIds(
  ids: string[],
): Promise<Event[]> {
  if (!ids.length) return [];

  const CHUNK_SIZE = 900;
  const results: Event[] = [];
  let c = 0;

  for (let i = 0; i < ids.length; i += CHUNK_SIZE) {
    const chunk = ids.slice(i, i + CHUNK_SIZE);

    const sql = `
    SELECT
      event_id, vector_id, token, token_idx, doc_id,
      pub_year, window_id, window_token_pos
    FROM events
    WHERE event_id IN (${ chunk.map(id => `'${ id }'`).join(",") });
  `;

    const rows = await execRows(sql);

    for (const r of rows) {
      results.push({
        event_id: String(r[0]),
        vector_id: String(r[1]),
        token: String(r[2]),
        token_idx: Number(r[3]) as number,
        doc_id: String(r[4]),
        pub_year: Number(r[5]),
        window_id: Number(r[6]),
        window_token_pos: Number(r[7]),
      });
    }
  }

  return results as Event[];
}

export async function queryEventsByConcept(
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

  const eventMap = new Map<string, ConceptEvent>();
  const events: ConceptEvent[] = [];

  for (const r of eventRows) {
    const e: ConceptEvent = {
      event_id: r[0] as string,
      vector_id: r[1] as string,
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
      event_id: String(r[1]),
      vector_id: r[2] != null ? String(r[2]) : undefined,
      token: r[3] as string,
      doc_id: r[4] as string,
      pub_year: r[5] as number,
      token_idx: String(r[6]),
      window_id: r[7] as number,
      window_token_pos: r[8] as number,
      score: r[9] as number,
    };
    eventMap.get(r[0] as string)?.neighbours.push(nb);
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
      string,
      number,
      string | null,
      string | null,
      number | null,
      number,
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


export async function queryYearCounts(
  concept: string,
): Promise<Map<number, number>> {
  const rows = await execRows(
    `SELECT pub_year, COUNT(*) as count
     FROM events
     WHERE concept = ?
       AND pub_year IS NOT NULL
     GROUP BY pub_year ORDER BY pub_year ASC`,
    [concept],
  );

  const map = new Map<number, number>();
  for (const [year, count] of rows as [number, number][]) {
    map.set(year, count);
  }
  return map;
}