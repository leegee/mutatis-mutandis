import type { Plugin } from "vite";
import path from "path";
import fs from "fs";

export function serveOutJsonPlugin(rootDir: string): Plugin {
    return {
        name: "vite-serve-out-json",
        configureServer(server) {
            server.middlewares.use(
                (req, res, next) => {
                    const match = req.url?.match(/^\/(api|xml)\/(.+)/);
                    let use, filePath;
                    if (match) {
                        use = match[1] === 'api' ? 'out' : 'eebo_all';
                        filePath = path.resolve(rootDir, use, match[2]);
                        if (fs.existsSync(filePath)) {
                            res.setHeader("Content-Type", "text/xml");
                            fs.createReadStream(filePath).pipe(res);
                            return;
                        }

                        else {
                            res.statusCode = 404;
                            res.setHeader("Content-Type", "application/json");
                            res.end(
                                JSON.stringify(
                                    {
                                        error: "File not found",
                                        url: req.url,
                                        m: match,
                                        rootDir,
                                        use, filePath
                                    },
                                    null,
                                    4
                                )
                            );
                            return;
                        }
                    }
                    next();
                }
            );
        },
    };
}
