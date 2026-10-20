import type { Plugin } from "vite";
import { Pool } from "pg";

import { createWindowMiddleware } from "../server-middleware/api/window.middleware";
import { createWindowBatchMiddleware } from "../server-middleware/api/window-batch.middleware";
import { createDocumentMiddleware } from "../server-middleware/api/doc.middleware";
import { createGroqMiddleware } from "../server-middleware/api/groq.middleware";
import { createStaticMiddleware } from "../server-middleware/static.middleware";
import { CORPUS_PG_DB_NAME } from "../src/corpus_config";

export function temporaryMiddlewarePlugin(rootDir: string): Plugin {
	const pool = new Pool({
		database: CORPUS_PG_DB_NAME,
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
