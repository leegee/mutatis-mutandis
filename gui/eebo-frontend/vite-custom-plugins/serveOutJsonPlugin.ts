import type { Plugin } from "vite";
import path from "path";
import fs from "fs";

export function serveOutJsonPlugin(rootDir: string): Plugin {
    return {
        name: "vite-serve-out-json",

        configureServer(server) {
            server.middlewares.use((req, res, next) => {
                if (!req.url) return next();

                const match = req.url.match(/^\/(api|xml)\/(.+)$/);
                if (!match) return next();

                const type = match[1];
                const rawPath = match[2];

                // Map route → folder
                const folder = type === "api" ? "out" : "eebo_all";

                // Normalize path:
                // 1. remove leading slashes
                // 2. prevent weird URL encoding issues
                const safeRelativePath = decodeURIComponent(rawPath)
                    .replace(/^\/+/, "");

                // Build final path safely
                const filePath = path.join(rootDir, folder, safeRelativePath);

                // SECURITY: ensure path stays inside rootDir
                const normalizedRoot = path.resolve(rootDir);
                const normalizedFile = path.resolve(filePath);

                if (!normalizedFile.startsWith(normalizedRoot)) {
                    res.statusCode = 403;
                    res.setHeader("Content-Type", "application/json");
                    res.end(JSON.stringify({
                        error: "Forbidden path traversal attempt",
                        url: req.url,
                        filePath
                    }, null, 4));
                    return;
                }

                // Check file existence
                if (fs.existsSync(normalizedFile)) {
                    res.setHeader("Content-Type", "text/xml");
                    fs.createReadStream(normalizedFile).pipe(res);
                    return;
                }

                // Not found
                res.statusCode = 404;
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify({
                    error: "File not found",
                    url: req.url,
                    filePath: normalizedFile
                }, null, 4));
            });
        },
    };
}