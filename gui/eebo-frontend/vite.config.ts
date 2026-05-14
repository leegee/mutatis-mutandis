import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import path from "path";
import { serveOutJsonPlugin } from "./vite-custom-plugins/serveOutJsonPlugin";

const documentRoot = path.resolve(__dirname, "../..");

console.info(`[vite.config] documentRoot: ${ documentRoot }`);

export default defineConfig({
  plugins: [
    solid(),
    serveOutJsonPlugin(documentRoot),
  ],
  server: {
    fs: {
      allow: [
        path.resolve(__dirname)
      ],
    },
  },
});