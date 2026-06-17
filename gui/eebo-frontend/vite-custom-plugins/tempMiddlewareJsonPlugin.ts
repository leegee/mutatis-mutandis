import type { Plugin } from "vite";
import { Pool } from "pg";
import { createWindowMiddleware } from "../server-middleware/api/window";
import { createWindowBatchMiddleware } from "../server-middleware/api/window-batch";
import { createDocumentMiddleware } from "../server-middleware/api/doc";
import { createGroqMiddleware } from "../server-middleware/api/groq";
import { createStaticMiddleware } from "../server-middleware/static";


export function temporaryMiddlewarePlugin(rootDir: string): Plugin {
  const pool = new Pool({
    database: "eebo",
  });

  return {
    name: "vite-serve-out-json",

    configureServer(server) {
      server.middlewares.use(createGroqMiddleware());
      server.middlewares.use(createWindowBatchMiddleware(pool));
      server.middlewares.use(createWindowMiddleware(pool));
      server.middlewares.use(createDocumentMiddleware(pool));
      server.middlewares.use(createStaticMiddleware(rootDir));
    },
  };
}
