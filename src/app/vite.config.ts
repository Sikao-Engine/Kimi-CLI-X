import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  server: {
    port: 5173,
    open: true,
  },
});
