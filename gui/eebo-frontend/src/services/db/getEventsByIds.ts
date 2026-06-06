import type { Event } from "../../types";
import { execRows } from "./dbh";

export async function getEventsByIds(
  ids: number[],
): Promise<Event[]> {
  if (!ids.length) return [];

  const CHUNK_SIZE = 900;
  const results: Event[] = [];
  let c = 0;

  console.log('[getEventsByIds] enter', ids)

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
        event_id: r[0] as string,
        vector_id: r[1] as string,
        token: r[2] as string,
        token_idx: r[3] as number,
        doc_id: r[4] as string,
        pub_year: r[5] as number,
        window_id: r[6] as number,
        window_token_pos: r[7] as number,
      });
    }
  }

  console.log('[getEventsByIds]', c, results.length)
  return results as Event[];
}
