import { controls } from "../../state/controls.store";
import type { EventQuery, SqliteEvent, SqliteEventWithNeighbours, SqliteNeighbour } from "../../types";
import { execRows } from "./dbh";

// stay safely under SQLite's ~999 bound-parameter limit
const SQLITE_MAX_VARIABLES = 900;

// ---------------------------------------------------------------------------
// Shared SQL fragments
//
// These were being hand-written slightly differently in every function
// (different aliases, JOIN vs LEFT JOIN, sometimes missing `role = 'seed'`).
// Centralising them means a fix/typo only needs to happen once.
// ---------------------------------------------------------------------------

/**
 * `events` has no `concept` column of its own -- concept membership lives on
 * concept_field_events (role='seed'). Always joins against an `events`
 * alias called `e`.
 */
function seedJoinSql(kind: "JOIN" | "LEFT JOIN" = "JOIN"): string {
  return `${ kind } concept_field_events f
              ON f.event_id = e.event_id
              AND f.role = 'seed'`;
}

/**
 * documents' PK is the composite (corpus, doc_id), so doc_id alone can
 * collide across corpora -- always join on both. Assumes an `events` alias
 * called `e`.
 */
function documentsJoinSql(alias = "d"): string {
  return `LEFT JOIN documents ${ alias }
              ON ${ alias }.doc_id = e.doc_id
              AND ${ alias }.corpus = e.corpus`;
}

/** `?, ?, ?, ...` for n placeholders. */
function placeholders(n: number): string {
  return Array(n).fill("?").join(",");
}

/** Split an array into chunks no larger than SQLITE_MAX_VARIABLES. */
function chunkArray<T>(arr: T[], size = SQLITE_MAX_VARIABLES): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

/**
 * The 9 columns nearly every event row needs, in a fixed order, so a single
 * mapper (below) can be reused instead of hand-writing the same
 * null-coalescing block per query. Callers select these 9 columns first
 * (via coreEventSelectList) and append anything extra (concept, author,
 * pub_place, score, ...) after them.
 */
function coreEventSelectList(alias = "e"): string {
  return [
    "event_id", "vector_id", "token", "doc_id", "pub_year",
    "token_idx", "window_id", "window_token_pos", "corpus",
  ].map((c) => `${ alias }.${ c }`).join(", ");
}

function mapCoreEventFields(r: any[], offset = 0) {
  return {
    event_id: String(r[offset + 0]),
    vector_id: r[offset + 1] != null ? String(r[offset + 1]) : null,
    token: (r[offset + 2] as string) ?? null,
    doc_id: (r[offset + 3] as string) ?? null,
    pub_year: r[offset + 4] != null ? Number(r[offset + 4]) : null,
    token_idx: r[offset + 5] != null ? Number(r[offset + 5]) : null,
    window_id: r[offset + 6] != null ? Number(r[offset + 6]) : null,
    window_token_pos: r[offset + 7] != null ? Number(r[offset + 7]) : null,
    corpus: r[offset + 8] != null ? String(r[offset + 8]) : null,
  };
}

// Width of the block mapCoreEventFields reads, so callers know where their
// own extra columns start (e.g. `const authorIdx = CORE_EVENT_WIDTH + 0`).
const CORE_EVENT_WIDTH = 9;


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

  let sql: string;
  let args: any[];

  if (concepts.length) {
    sql = `SELECT
          e.event_id,
          e.doc_id,
          e.pub_year,
          e.token,
          e.corpus
      FROM events e
          ${ seedJoinSql() }
      WHERE e.pub_year BETWEEN ? AND ?
        AND f.concept IN (${ placeholders(concepts.length) })
      `;

    args = [controls.fromYear, controls.toYear, ...concepts];
  } else {
    // No concept filter selected -- return all events, any concept.
    sql = `SELECT
          e.event_id,
          e.doc_id,
          e.pub_year,
          e.token,
          e.corpus
      FROM events e
      WHERE e.pub_year BETWEEN ? AND ?
      `;

    args = [controls.fromYear, controls.toYear];
  }

  console.debug("[queries.fetchEvents] " + sql, args)

  const rows = await execRows(sql, args);

  const parsedRows = rows
    .map((r) => {
      return {
        event_id: r[0],
        doc_id: r[1],
        pub_year: r[2],
        token: r[3],
        corpus: r[4],
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
  corpus: string;
  pub_place: string | null;
  normalized_place: string | null;
  lat: number | null;
  lng: number | null;
};

export async function fetchEventsGeo(): Promise<EventGeoRow[]> {
  const concepts = controls.conceptSelection ?? [];

  const conceptJoin = concepts.length
    ? `${ seedJoinSql() }
          AND f.concept IN (${ placeholders(concepts.length) })`
    : "";

  // ASSUMPTION: document_places is keyed like documents, i.e.
  // (corpus, doc_id) -- unconfirmed, since it's not in the schema shared so
  // far. If it's actually keyed on doc_id alone, drop the
  // `document_places.corpus = e.corpus` condition.
  const sql = `
    SELECT
        e.event_id,
        e.doc_id,
        e.pub_year,
        e.token,
        e.corpus,
        documents.pub_place,
        document_places.normalized_place,
        document_places.lat,
        document_places.lng
    FROM events e
    ${ conceptJoin }
    ${ documentsJoinSql("documents") }
    LEFT JOIN document_places
        ON document_places.doc_id = e.doc_id
        AND document_places.corpus = e.corpus
    WHERE e.pub_year BETWEEN ? AND ?
  `;

  const args = [
    ...concepts,
    controls.fromYear,
    controls.toYear,
  ];

  console.debug("[queries.fetchEventsGeo]", sql, args);

  const rows = await execRows(sql, args);

  const parsedRows: EventGeoRow[] = rows.map((r) => ({
    event_id: String(r[0]),
    doc_id: String(r[1]),
    pub_year: Number(r[2]),
    token: String(r[3]),
    corpus: String(r[4]),
    pub_place: r[5] != null ? String(r[5]) : null,
    normalized_place: r[6] != null ? String(r[6]) : null,
    lat: r[7] != null ? Number(r[7]) : null,
    lng: r[8] != null ? Number(r[8]) : null,
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
    `SELECT MIN(e.pub_year), MAX(e.pub_year)
     FROM   events e
         ${ seedJoinSql() }
     WHERE  f.concept = ?
       AND  e.pub_year IS NOT NULL`,
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

  const eventRows = await execRows(
    `SELECT
        ${ coreEventSelectList() },
        d.author, d.pub_place, f.concept
     FROM events e
         ${ seedJoinSql("LEFT JOIN") }
         ${ documentsJoinSql() }
     WHERE e.event_id = ?
     LIMIT 1`,
    [id],
  );

  const eventRow = eventRows[0];
  // console.trace("[queryEventById]", eventRow)

  if (eventRow) {
    return {
      type: 'event',
      ...mapCoreEventFields(eventRow),
      author: eventRow[CORE_EVENT_WIDTH + 0] != null ? eventRow[CORE_EVENT_WIDTH + 0] : null,
      pub_place: eventRow[CORE_EVENT_WIDTH + 1] != null ? eventRow[CORE_EVENT_WIDTH + 1] : null,
      concept: eventRow[CORE_EVENT_WIDTH + 2] != null ? String(eventRow[CORE_EVENT_WIDTH + 2]) : null,
    } as SqliteEvent;
  }

  // Not a seed event -- check whether it's a neighbour instead. `score` is
  // appended after the core 9 + author/pub_place here (rather than sitting
  // in the middle, as it originally did), specifically so this row shares
  // mapCoreEventFields with the event branch above instead of needing its
  // own hand-rolled mapping -- which is what let `score` and `concept`
  // collide on the same index in the first place.
  const neighbourRows = await execRows(
    `SELECT
        ${ coreEventSelectList() },
        d.author, d.pub_place, n.score
     FROM neighbours n
         JOIN events e
             ON e.event_id = n.neighbour_event_id
         ${ documentsJoinSql() }
     WHERE n.neighbour_event_id = ?
     LIMIT 1`,
    [id],
  );

  const neighbourRow = neighbourRows[0];
  if (!neighbourRow) return null;

  return {
    type: 'neighbour',
    ...mapCoreEventFields(neighbourRow),
    author: neighbourRow[CORE_EVENT_WIDTH + 0] != null ? neighbourRow[CORE_EVENT_WIDTH + 0] : null,
    pub_place: neighbourRow[CORE_EVENT_WIDTH + 1] != null ? neighbourRow[CORE_EVENT_WIDTH + 1] : null,
    score: Number(neighbourRow[CORE_EVENT_WIDTH + 2]),
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

  for (const chunk of chunkArray(uniqueIds)) {
    const rows = await execRows(
      `SELECT
          ${ coreEventSelectList() },
          f.concept
       FROM events e
           ${ seedJoinSql("LEFT JOIN") }
       WHERE e.event_id IN (${ placeholders(chunk.length) })`,
      chunk,
    );

    for (const row of rows) {
      const event: SqliteEvent = {
        ...mapCoreEventFields(row),
        concept: row[CORE_EVENT_WIDTH + 0] as string,
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

  const results: SqliteEvent[] = [];

  for (const chunk of chunkArray(ids)) {
    // Was building this with raw string interpolation (`'${id}'`) -- a SQL
    // injection risk the original comment already flagged ("Should escape
    // that"). Now uses parameter binding + the shared chunking/select
    // helpers, same as queryEventsByIds above (these two functions were
    // near-duplicates of each other with slightly different chunk sizes
    // and column orders).
    const sql = `
    SELECT
      ${ coreEventSelectList() },
      f.concept
    FROM events e
        ${ seedJoinSql("LEFT JOIN") }
    WHERE e.event_id IN (${ placeholders(chunk.length) })
  `;

    const rows = await execRows(sql, chunk);

    for (const r of rows) {
      results.push({
        ...mapCoreEventFields(r),
        concept: String(r[CORE_EVENT_WIDTH + 0]),
      } as SqliteEvent);
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
    `SELECT ${ coreEventSelectList() }
     FROM   events e
         ${ seedJoinSql() }
     WHERE  f.concept  = ?
       AND  e.pub_year >= ?
       AND  e.pub_year <= ?
     ORDER  BY e.pub_year, e.event_id`,
    [concept, fromYear, toYear],
  );

  if (eventRows.length === 0) return [];

  const eventMap = new Map<string, SqliteEventWithNeighbours>();
  const events: SqliteEventWithNeighbours[] = [];

  for (const r of eventRows) {
    const e: SqliteEventWithNeighbours = {
      ...mapCoreEventFields(r),
      concept,
      neighbours: [],
    } as SqliteEventWithNeighbours;
    eventMap.set(e.event_id, e);
    events.push(e);
  }

  const ids = [...eventMap.keys()].join(",");
  const nbRows = await execRows(
    `SELECT
      n.event_id,
      ${ coreEventSelectList() },
      n.score
   FROM neighbours n
   JOIN events e
       ON e.event_id = n.neighbour_event_id
   WHERE n.event_id IN (${ ids })
   ORDER BY n.event_id, n.score DESC`,
  );

  for (const r of nbRows) {
    // r[0] is the *seed* event_id (for keying eventMap), so the core-9
    // block for the neighbour itself starts at offset 1.
    const nb: SqliteNeighbour = {
      ...mapCoreEventFields(r, 1),
      score: Number(r[1 + CORE_EVENT_WIDTH]),
    } as SqliteNeighbour;

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
  // NOTE: this aggregates across all corpora for the given concept/year. If
  // you want counts broken out per corpus, the return type needs to change
  // (e.g. Map<string, Map<number, number>>) since a plain Map<year, count>
  // can't represent more than one corpus per year.
  const rows = await execRows(
    `SELECT e.pub_year, COUNT(*) as count
     FROM events e
         ${ seedJoinSql() }
     WHERE f.concept = ?
       AND e.pub_year IS NOT NULL
     GROUP BY e.pub_year ORDER BY e.pub_year ASC`,
    [concept],
  );

  const map = new Map<number, number>();
  for (const [year, count] of rows as [number, number][]) {
    map.set(year, count);
  }
  return map;
}
