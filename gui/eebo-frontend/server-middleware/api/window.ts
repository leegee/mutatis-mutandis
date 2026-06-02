import type { Connect } from "vite";
import { Pool } from "pg";
import { serverError, text } from "../lib/middleware";

const TOKEN_WINDOW_HALF = 30;

// /api/window/:docId/:tokenId

export function createWindowMiddleware(pool: Pool): Connect.NextHandleFunction {
  return async (req, res, next) => {
    if (!req.url) return next();

    const match = req.url.match(/^\/api\/window\/([^/]+)\/([^/]+)$/);

    if (!match) return next();

    const docId = match[1];
    const tokenId = Number(match[2]);

    try {
      if (!Number.isInteger(tokenId)) {
        throw new RangeError("Invalid tokenId");
      }

      const result = await pool.query(
        `
                SELECT token, token_idx
                FROM pamphlet_tokens
                WHERE doc_id = $1
                AND token_idx BETWEEN ($2::int - $3) AND ($2::int + $3)
                ORDER BY token_idx
                `,
        [docId, tokenId, TOKEN_WINDOW_HALF],
      );

      const content = result.rows
        .map((row) =>
          row.token_idx === tokenId ? `<mark>${row.token}</mark>` : row.token,
        )
        .join(" ");

      console.log(`[api/window] ${docId}/${tokenId}`);

      return text(res, 200, content);
    } catch (error) {
      return serverError(res, error);
    }
  };
}
