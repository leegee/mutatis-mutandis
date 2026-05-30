import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import path from "path";
import { temporaryMiddlewarePlugin } from "./vite-custom-plugins/tempMiddlewareJsonPlugin";

const documentRoot = path.resolve(__dirname, "../..");

console.info(`[vite.config] documentRoot: ${ documentRoot }`);

export default defineConfig({
  plugins: [
    solid(),
    temporaryMiddlewarePlugin(documentRoot),
  ],
  server: {
    fs: {
      allow: [
        path.resolve(__dirname)
      ],
    },
  },
});