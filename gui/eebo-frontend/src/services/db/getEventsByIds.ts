// import type { Event } from "../../types";
// import { execRows } from "./dbh";

// export async function getEventsByIds(
//   ids: string[],
// ): Promise<Event[]> {
//   if (!ids.length) return [];

//   const CHUNK_SIZE = 900;
//   const results: Event[] = [];
//   let c = 0;

//   for (let i = 0; i < ids.length; i += CHUNK_SIZE) {
//     const chunk = ids.slice(i, i + CHUNK_SIZE);

//     const sql = `
//     SELECT
//       event_id, vector_id, token, token_idx, doc_id,
//       pub_year, window_id, window_token_pos
//     FROM events
//     WHERE event_id IN (${ chunk.map(id => `'${ id }'`).join(",") });
//   `;

//     const rows = await execRows(sql);

//     for (const r of rows) {
//       results.push({
//         event_id: String(r[0]),
//         vector_id: String(r[1]),
//         token: String(r[2]),
//         token_idx: Number(r[3]) as number,
//         doc_id: String(r[4]),
//         pub_year: Number(r[5]),
//         window_id: Number(r[6]),
//         window_token_pos: Number(r[7]),
//       });
//     }
//   }

//   return results as Event[];
// }
