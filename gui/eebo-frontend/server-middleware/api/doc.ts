import type { Connect } from "vite";
import { Pool } from "pg";
import { json, serverError, redirect } from "../lib/middleware";

export function createDocumentMiddleware(
  pool: Pool,
): Connect.NextHandleFunction {
  return async (req, res, next) => {
    if (!req.url) return next();
    const match = req.url.match(/^\/api\/doc\/([^/]+)$/);
    if (!match) return next();
    const docId = match[1];

    try {
      const result = await pool.query(
        `
                SELECT *
                FROM documents
                WHERE doc_id = $1
                `,
        [docId],
      );

      const row = result.rows[0];

      if (!row) {
        return json(res, 404, {
          error: "Document not found",
          docId,
        });
      }

      if (!row.filepath) {
        throw new Error('No filepath in db for doc ' + docId)
      }

      const filepath: string = row.filepath;
      console.log(filepath)
      const matchPath = filepath.match(/eebo_all[\\/](.+)$/);

      if (!matchPath) {
        return json(res, 500, {
          error: "Invalid filepath format",
          filepath,
        });
      }

      const relativePath = matchPath[1].replace(/\\/g, "/");

      const redirectUrl = `/xml/${ relativePath }`;

      console.log(`[api/doc] ${ docId } -> ${ redirectUrl } for ${ req.url }`);

      return redirect(res, redirectUrl);
    } catch (error) {
      return serverError(res, error);
    }
  };
}
