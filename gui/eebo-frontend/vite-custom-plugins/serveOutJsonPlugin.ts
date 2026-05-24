import type { Plugin } from "vite";
import path from "path";
import fs from "fs";
import { Pool } from "pg";

export function serveOutJsonPlugin(rootDir: string): Plugin {
    // Postgres uses default env vars:
    // PGHOST, PGPORT, PGUSER, PGPASSWORD
    const pool = new Pool({
        database: "eebo",
    });

    return {
        name: "vite-serve-out-json",

        configureServer(server) {
            server.middlewares.use(async (req, res, next) => {
                if (!req.url) return next();

                /**
                 * =========================
                 * API: /api/doc/:id
                 * =========================
                 */
                const matchApiDoc = req.url.match(/^\/api\/doc\/(.+)$/);

                if (matchApiDoc) {
                    const docId = matchApiDoc[1];

                    try {
                        const result = await pool.query(
                            "SELECT * FROM documents WHERE doc_id = $1",
                            [docId]
                        );

                        const row = result.rows[0];

                        if (!row) {
                            res.statusCode = 404;
                            res.setHeader("Content-Type", "application/json");

                            res.end(JSON.stringify({
                                error: "Document not found",
                                docId,
                            }, null, 2));

                            return;
                        }

                        const filepath: string = row.filepath;

                        // Extract everything after "eebo_all/"
                        const matchPath = filepath.match(/eebo_all[\\/](.+)$/);

                        if (!matchPath) {
                            res.statusCode = 500;
                            res.setHeader("Content-Type", "application/json");

                            res.end(JSON.stringify({
                                error: "Invalid filepath format",
                                filepath,
                            }, null, 2));

                            return;
                        }

                        const relativePath = matchPath[1].replace(/\\/g, "/");

                        const redirectUrl = `/xml/${ relativePath }`;

                        console.log(`[api/doc] ${ docId } -> ${ redirectUrl }`);

                        // Redirect to XML route
                        res.statusCode = 302;
                        res.setHeader("Location", redirectUrl);
                        res.end();

                        return;

                    } catch (err: any) {
                        console.error("Postgres error:", err);

                        res.statusCode = 500;
                        res.setHeader("Content-Type", "application/json");

                        res.end(JSON.stringify({
                            error: err.message,
                        }, null, 2));

                        return;
                    }
                }

                /**
                 * =========================
                 * STATIC FILE ROUTES
                 * /json/* and /xml/*
                 * =========================
                 */
                const matchStatic = req.url.match(/^\/(json|xml)\/(.+)$/);

                if (!matchStatic) return next();

                const type = matchStatic[1];
                const rawPath = matchStatic[2];

                const folder = type === "json" ? "out" : "eebo_all";

                const safeRelativePath = decodeURIComponent(rawPath)
                    .replace(/^\/+/, "");

                const filePath = path.join(rootDir, folder, safeRelativePath);

                console.log(`[static] ${ req.url } -> ${ filePath }`);

                const normalizedRoot = path.resolve(rootDir);
                const normalizedFile = path.resolve(filePath);

                // prevent path traversal
                if (!normalizedFile.startsWith(normalizedRoot)) {
                    res.statusCode = 403;
                    res.setHeader("Content-Type", "application/json");

                    res.end(JSON.stringify({
                        error: "Forbidden path traversal attempt",
                        url: req.url,
                    }, null, 2));

                    return;
                }

                // serve XML/JSON files
                if (fs.existsSync(normalizedFile)) {
                    res.setHeader(
                        "Content-Type",
                        type === "json" ? "application/json" : "text/xml"
                    );

                    fs.createReadStream(normalizedFile).pipe(res);
                    return;
                }

                res.statusCode = 404;
                res.setHeader("Content-Type", "application/json");

                res.end(JSON.stringify({
                    error: "File not found",
                    url: req.url,
                }, null, 2));
            });
        },
    };
}