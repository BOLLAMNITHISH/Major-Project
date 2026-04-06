import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["plotly.js-dist-min"],
  },
  server: {
    port: 5173,
    proxy: {
      "/predict": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/explain": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/evaluation": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/history": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
