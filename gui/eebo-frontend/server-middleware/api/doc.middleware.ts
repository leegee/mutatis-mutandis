import type { Connect } from "vite";
import { Pool } from "pg";
import { json, serverError, redirect } from "../lib/response";

export function createDocumentMiddleware(
  pool: Pool,
): Connect.NextHandleFunction {
  return async (req, res, next) => {
    if (!req.url) return next();
    const match = req.url.match(/^\/api\/+doc\/+([^/]+)\/+([^/]+)$/);
    if (!match) return next();
    const corpus = match[2];
    const docId = match[2];

    try {
      const result = await pool.query(
        `
                SELECT *
                FROM documents
                WHERE corpus = $1 AND doc_id = $2
                `,
        [corpus, docId],
      );

      const row = result.rows[0];

      if (!row) {
        return json(res, 404, {
          error: "Document not found",
          corpus, docId,
        });
      }

      if (!row.filepath) {
        throw new Error(`No filepath in db for corpus/doc ${ corpus }/${ docId }`)
      }

      const filepath: string = row.filepath;
      const redirectUrl = `/xml/${ filepath }`;
      console.log(`[api/doc] ${ docId } -> ${ redirectUrl } for ${ req.url }`);
      return redirect(res, redirectUrl);
    }

    catch (error) {
      return serverError(res, error);
    }
  };
}
