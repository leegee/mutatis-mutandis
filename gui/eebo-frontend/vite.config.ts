import fs from "node:fs";

import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import path from "path";
import { temporaryMiddlewarePlugin } from "./vite-custom-plugins/tempMiddlewareJsonPlugin";

const documentRoot = path.resolve(__dirname, "../..");

console.info(`[vite.config] documentRoot: ${ documentRoot }`);

export default defineConfig({
  optimizeDeps: {
    exclude: ["@sqlite.org/sqlite-wasm"],
  },
  plugins: [
    solid(),
    temporaryMiddlewarePlugin(documentRoot),
  ],
  server: {
    port: 3443,
    https: {
      key: fs.readFileSync("certs/key.pem"),
      cert: fs.readFileSync("certs/cert.pem"),
    },

    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: {
      allow: [
        path.resolve(__dirname)
      ],
    },
  },
});