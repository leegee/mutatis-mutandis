import type { Plugin } from "vite";
import path from "path";
import fs from "fs";

export function serveOutJsonPlugin(outDir: string): Plugin {
    return {
        name: "vite-serve-out-json",
        configureServer(server) {
            server.middlewares.use((req, res, next) => {
                const match = req.url?.match(/^\/(api|xml)\/(.+)/);
                if (match) {
                    const filePath = path.resolve(outDir, match[2]);
                    if (fs.existsSync(filePath)) {
                        res.setHeader("Content-Type", "application/json");
                        fs.createReadStream(filePath).pipe(res);
                        return;
                    }

                    else {
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
