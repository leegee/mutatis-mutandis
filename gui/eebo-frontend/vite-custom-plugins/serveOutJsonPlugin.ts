import { Plugin } from "vite";
import path from "path";
import fs from "fs";

export function serveOutJsonPlugin(outDir: string): Plugin {
    return {
        name: "vite-serve-out-json",
        configureServer(server) {
            server.middlewares.use((req, res, next) => {
                if (req.url?.startsWith("/api/")) {
                    // Map /api/foo.json → ../out/foo.json
                    const filePath = path.resolve(outDir, req.url.replace("/api/", ""));
                    if (fs.existsSync(filePath)) {
                        res.setHeader("Content-Type", "application/json");
                        fs.createReadStream(filePath).pipe(res);
                        return;
                    } else {
                        res.statusCode = 404;
                        res.end(JSON.stringify({ error: "File not found" }));
                        return;
                    }
                }
                next();
            });
        },
    };
}
