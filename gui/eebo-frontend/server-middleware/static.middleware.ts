import type { Connect } from "vite";
import path from "path";
import fs from "fs";
import { json } from "./lib/response";

export function createStaticMiddleware(
  rootDir: string,
): Connect.NextHandleFunction {
  return (req, res, next) => {
    if (!req.url) return next();

    const match = req.url.match(/^\/(json|xml)\/(.+)$/);
    if (!match) return next();

    const [, type, rawPath] = match;

    const dir = type === "json" ? "out" : "eebo_all";
    const relativePath = decodeURIComponent(rawPath).replace(/^\/+/, "");
    const filePath = path.resolve(rootDir, dir, relativePath);
    const rootPath = path.resolve(rootDir);

    console.log(`[static] ${ req.url } -> ${ filePath }`);

    if (!filePath.startsWith(rootPath)) {
      return json(res, 403, {
        error: "Forbidden path traversal attempt",
        url: req.url,
      });
    }

    if (!fs.existsSync(filePath)) {
      return json(res, 404, {
        error: "File not found",
        url: req.url,
      });
    }

    if (type === "xml") {
      res.setHeader("Content-Type", "application/xml; charset=utf-8");

      const stream = fs.createReadStream(filePath, { encoding: "utf8" });

      let buffer = "";

      stream.on("data", (chunk) => {
        buffer += chunk;

        // prevent unbounded growth in case of very large files
        if (buffer.length > 1_000_000) {
          res.write(buffer);
          buffer = "";
        }
      });

      stream.on("end", () => {
        const rewritten = buffer.replaceAll(
          "http://www.textcreationpartnership.org/docs/code/pfs.css",
          "/xml-styles/pfs.css",
        );

        res.end(rewritten);
      });

      stream.on("error", (err) => {
        console.error(err);
        res.statusCode = 500;
        res.end("Internal Server Error");
      });

      return;
    }

    // JSON path (unchanged)
    res.setHeader("Content-Type", "application/json");
    fs.createReadStream(filePath).pipe(res);
  };
}
