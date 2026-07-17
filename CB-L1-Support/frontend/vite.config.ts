import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

/**
 * The production build is emitted straight into the FastAPI static directory so
 * `python run_chatbot.py --serve` ships the compiled SPA. During development,
 * `/api` and `/health` are proxied to the FastAPI server on :5100, preserving
 * the exact same-origin contract used in production.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:5100", changeOrigin: true },
      "/health": { target: "http://localhost:5100", changeOrigin: true },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../backend/api/static"),
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
});
