/**
 * /api/window/batch
 *
 * TODO Share types with client
 *
 * IN:
 *
 * {
 *  "queries": [
 *     { "corpus": "eebo", "docId": "a", "tokenIdx": 12 },
 *   ]
 * }
 *
 * OUT:
 * {
 *  "results": [
 *    {"corpus": "eebo",  "docId": "A", "tokenIdx": 12, "content": "...", token: "T" },
 *  ]
 * }
 */
import type { Connect } from "vite";
import { Pool } from "pg";
import { serverError, text } from "../lib/response";

const TOKEN_WINDOW_HALF = 10;

type QueryItem = {
  corpus: string;
  docId: string;
  tokenIdx: number;
};

function format(text: string) {
  return text.replace(/\s+([,.;:\)])/g, "$1").replace(/\(\s+/g, "(");
}

export function createWindowBatchMiddleware(pool: Pool): Connect.NextHandleFunction {
  return async (req, res, next) => {
    if (req.method !== "POST") return next();
    if (!req.url || req.url !== "/api/window/batch") return next();


    try {
      const body = await new Promise<string>((resolve, reject) => {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => resolve(data));
        req.on("error", reject);
      });

      const parsed = JSON.parse(body);
      console.log(req.url, body);

      if (!parsed || !Array.isArray(parsed.queries)) {
        throw new TypeError("queries must be an array");
      }

      const queries: QueryItem[] = parsed.queries.map((q: any) => ({
        corpus: String(q.corpus),
        docId: String(q.docId),
        tokenIdx: Number(q.tokenIdx),
      }));

      if (queries.some(q => !Number.isInteger(q.tokenIdx))) {
        throw new RangeError("Invalid tokenIdx in queries");
      }

      // group by docId (optimization only)
      const grouped = new Map<string, QueryItem[]>();

      for (const q of queries) {
        if (!grouped.has(q.docId)) grouped.set(q.docId, []);
        grouped.get(q.docId)!.push(q);
      }

      const results: any[] = [];

      // per-doc batch query
      for (const [docId, items] of grouped.entries()) {
        const corpus = items[0].corpus;
        const tokenIds = items.map(i => i.tokenIdx);

        const min = Math.min(...tokenIds) - TOKEN_WINDOW_HALF;
        const max = Math.max(...tokenIds) + TOKEN_WINDOW_HALF;

        const result = await pool.query(
          `
          SELECT token, token_idx
          FROM pamphlet_tokens
          WHERE corpus = $1
            AND doc_id = $2
            AND token_idx BETWEEN $3 AND $4
          ORDER BY token_idx
          `,
          [corpus, docId, min, max],
        );

        // slice out each item's own window and mark only its own token
        for (const item of items) {
          const lo = item.tokenIdx - TOKEN_WINDOW_HALF;
          const hi = item.tokenIdx + TOKEN_WINDOW_HALF;

          const content = result.rows
            .filter((row) => row.token_idx >= lo && row.token_idx <= hi)
            .map((row) =>
              row.token_idx === item.tokenIdx
                ? `<mark>${ row.token }</mark>`
                : row.token,
            )
            .join(" ");

          results.push({
            corpus,
            docId,
            tokenIdx: item.tokenIdx,
            content: format(content),
            token: result.rows.find(r => r.token_idx === item.tokenIdx)?.token
          });
        }
      }

      // console.debug(`[api/window:batch] returning ${ results.length } results`);
      return text(res, 200, JSON.stringify({ results }));
    }
    catch (error) {
      return serverError(res, error);
    }
  };
}
