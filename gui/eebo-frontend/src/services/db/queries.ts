import { controls } from "../../state/controls.store";
import type { EventQuery, SqliteEvent, SqliteEventWithNeighbours, SqliteNeighbour } from "../../types";
import { execRows } from "./dbh";

const SQLITE_MAX_VARIABLES = 900; // stay safely under SQLite's c999 limit

export function buildEventQuery(): EventQuery {
  return {
    concept: controls.conceptSelection[0],
    fromYear: controls.fromYear,
    toYear: controls.toYear,
    selectedEventIds: controls.selectedEventIds
      ? Array.from(controls.selectedEventIds)
      : null,
  };
}

// No bbox yet
export async function fetchEvents() {
  const concepts = controls.conceptSelection ?? [];

  const conceptPlaceholders = concepts.length
    ? `IN (${ concepts.map(() => "?").join(",") })`
    : "IS NOT NULL";

  const sql = `SELECT
        event_id,
        doc_id,
        pub_year,
        token
    FROM events
    WHERE pub_year BETWEEN ? AND ?
    AND concept ${ conceptPlaceholders }
    `;

  const args = [controls.fromYear, controls.toYear, ...concepts];

  console.debug("[queries.fetchEvents] " + sql, args)

  const rows = await execRows(sql, args);

  const parsedRows = rows
    .map((r) => {
      return {
        event_id: r[0],
        doc_id: r[1],
        pub_year: r[2],
        token: r[3],
      };
    })
    .filter(Boolean);

  console.debug("[queries.fetchEvents] RV", [parsedRows])
  return parsedRows;
}



export type EventGeoRow = {
  event_id: string;
  doc_id: string;
  pub_year: number;
  token: string;
  pub_place: string | null;
  normalized_place: string | null;
  lat: number | null;
  lng: number | null;
};

export async function fetchEventsGeo(): Promise<EventGeoRow[]> {
  const concepts = controls.conceptSelection ?? [];

  const conceptPlaceholders = concepts.length
    ? `IN (${ concepts.map(() => "?").join(",") })`
    : "IS NOT NULL";

  const sql = `
    SELECT
        events.event_id,
        events.doc_id,
        events.pub_year,
        events.token,
        documents.pub_place,
        document_places.normalized_place,
        document_places.lat,
        document_places.lng
    FROM events
    LEFT JOIN documents
        ON events.doc_id = documents.doc_id
    LEFT JOIN document_places
        ON events.doc_id = document_places.doc_id
    WHERE events.pub_year BETWEEN ? AND ?
    AND concept ${ conceptPlaceholders }
  `;

  const args = [
    controls.fromYear,
    controls.toYear,
    ...concepts,
  ];

  console.debug("[queries.fetchEventsGeo]", sql, args);

  const rows = await execRows(sql, args);

  const parsedRows: EventGeoRow[] = rows.map((r) => ({
    event_id: String(r[0]),
    doc_id: String(r[1]),
    pub_year: Number(r[2]),
    token: String(r[3]),
    pub_place: String(r[4]),
    normalized_place: String(r[5]),
    lat: Number(r[6]),
    lng: Number(r[7]),
  }));

  console.debug("[queries.fetchEventsGeo] rows", parsedRows.length, parsedRows.slice(0, 5));
  return parsedRows;
}


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
  return [Number(row[0]), Number(row[1])];
}


export async function queryEventById(id: string): Promise<SqliteEvent | null> {
  // console.trace("[query] queryEventById", id);
  if (typeof id !== 'string') {
    console.error('queryEventById received', typeof id);
    return null;
  }

  let type = 'event';
  const eventRows = await execRows(
    `SELECT event_id, concept, vector_id, token, events.doc_id, events.pub_year, token_idx, window_id, window_token_pos,
    documents.author, documents.pub_place
    FROM events
    LEFT JOIN documents ON events.doc_id = documents.doc_id
    WHERE event_id = ? LIMIT 1`,
    [id],
  );

  let row = eventRows[0];
  // console.trace("[queryEventById]", row)

  if (!row) {
    type = 'neighbour';
    const neighbourRow = await execRows(`
      SELECT
        n.neighbour_event_id,
        n.score,
        e.vector_id,
        e.token,
        e.doc_id,
        e.pub_year,
        e.token_idx,
        e.window_id,
        e.window_token_pos,
        d.author,
        d.pub_place
    FROM neighbours n
    JOIN events e ON e.event_id = n.neighbour_event_id
    LEFT JOIN documents d ON e.doc_id = d.doc_id
    WHERE n.neighbour_event_id = ?
    LIMIT 1
      `,
      [id],
    );

    if (neighbourRow.length > 0) row = neighbourRow[0];
  }

  if (!row) return null;

  return {
    type,
    event_id: String(row[0]),
    score: Number(row[1]),
    vector_id: row[2] != null ? String(row[2]) : null,
    token: (row[3] as string) ?? null,
    doc_id: (row[4] as string) ?? null,
    pub_year: row[5] != null ? Number(row[5]) : null,
    token_idx: row[6] != null ? Number(row[6]) : null,
    window_id: row[7] != null ? Number(row[7]) : null,
    window_token_pos: row[8] != null ? Number(row[8]) : null,
    author: row[9] != null ? row[9] : null,
    pub_place: row[10] != null ? row[10] : null,
  } as SqliteNeighbour;
}


export async function queryEventsByIds(
  ids: string[],
): Promise<Map<string, SqliteEvent>> {
  const result = new Map<string, SqliteEvent>();

  // de-duplicate while preserving nothing in particular — order doesn't
  // matter since we return a Map
  const uniqueIds = Array.from(new Set(ids));
  if (uniqueIds.length === 0) return result;

  console.debug("[query] queryEventsByIds", uniqueIds.length);

  for (let i = 0; i < uniqueIds.length; i += SQLITE_MAX_VARIABLES) {
    const chunk = uniqueIds.slice(i, i + SQLITE_MAX_VARIABLES);
    const placeholders = chunk.map(() => "?").join(",");

    const rows = await execRows(
      `SELECT event_id, concept, vector_id, token, doc_id, pub_year, token_idx, window_id, window_token_pos
       FROM events
       WHERE event_id IN (${ placeholders })`,
      chunk,
    );

    for (const row of rows) {
      const event: SqliteEvent = {
        event_id: String(row[0]),
        concept: row[1] as string,
        vector_id: row[2] != null ? String(row[2]) : null,
        token: (row[3] as string) ?? null,
        doc_id: (row[4] as string) ?? null,
        pub_year: row[5] != null ? Number(row[5]) : null,
        token_idx: row[6] != null ? Number(row[6]) : null,
        window_id: row[7] != null ? Number(row[7]) : null,
        window_token_pos: row[8] != null ? Number(row[8]) : null,
      } as SqliteEvent;

      result.set(event.event_id, event);
    }
  }

  return result;
}


export async function getEventsByIds(
  ids: string[],
): Promise<SqliteEvent[]> {
  if (!ids.length) return [];

  const CHUNK_SIZE = 900;
  const results: SqliteEvent[] = [];

  for (let i = 0; i < ids.length; i += CHUNK_SIZE) {
    const chunk = ids.slice(i, i + CHUNK_SIZE);

    const sql = `
    SELECT
      event_id, vector_id, token, token_idx, doc_id,
      pub_year, window_id, window_token_pos, concept
    FROM events
    WHERE event_id IN (${ chunk.map(id => `'${ id }'`).join(",") });
  `; // Should escape that

    const rows = await execRows(sql);

    for (const r of rows) {
      results.push({
        event_id: String(r[0]),
        vector_id: String(r[1]),
        token: String(r[2]),
        token_idx: Number(r[3]),
        doc_id: String(r[4]),
        pub_year: Number(r[5]),
        window_id: Number(r[6]),
        window_token_pos: Number(r[7]),
        concept: String(r[8]),
      });
    }
  }

  return results as SqliteEvent[];
}

export async function queryEventsByConcept(
  concept: string,
  fromYear: number,
  toYear: number,
): Promise<SqliteEventWithNeighbours[]> {
  const eventRows = await execRows(
    `SELECT event_id, vector_id, token, doc_id, pub_year,
            token_idx, window_id, window_token_pos, concept
     FROM   events
     WHERE  concept   = ?
       AND  pub_year >= ?
       AND  pub_year <= ?
     ORDER  BY pub_year, event_id`,
    [concept, fromYear, toYear],
  );

  if (eventRows.length === 0) return [];

  const eventMap = new Map<string, SqliteEventWithNeighbours>();
  const events: SqliteEventWithNeighbours[] = [];

  for (const r of eventRows) {
    const e: SqliteEventWithNeighbours = {
      event_id: r[0] as string,
      vector_id: r[1] as string,
      token: r[2] as string,
      doc_id: r[3] as string,
      pub_year: Number(r[4]),
      token_idx: Number(r[5]),
      window_id: Number(r[6]),
      window_token_pos: Number(r[7]),
      concept: String(r[8]),
      neighbours: [],
    };
    eventMap.set(e.event_id, e);
    events.push(e);
  }

  const ids = [...eventMap.keys()].join(",");
  const nbRows = await execRows(
    `SELECT
      n.event_id,
      n.neighbour_event_id,
      e.vector_id,
      e.token,
      e.doc_id,
      e.pub_year,
      e.token_idx,
      e.window_id,
      e.window_token_pos,
      n.score
   FROM neighbours n
   JOIN events e
       ON e.event_id = n.neighbour_event_id
   WHERE n.event_id IN (${ ids })
   ORDER BY n.event_id, n.score DESC`,
  );

  for (const r of nbRows) {
    const nb: SqliteNeighbour = {
      event_id: String(r[1]),
      vector_id: String(r[2]),
      token: r[3] as string,
      doc_id: r[4] as string,
      pub_year: Number(r[5]),
      token_idx: Number(r[6]),
      window_id: Number(r[7]),
      window_token_pos: Number(r[8]),
      score: Number(r[9]),
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
  return (Number(rows[0]?.[0])) ?? 0;
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
