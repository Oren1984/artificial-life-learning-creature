import { defineConfig } from "vitest/config";

export default defineConfig({
  server: { proxy: { "/api": "http://localhost:8080" } },
  test: { environment: "jsdom", include: ["tests/**/*.test.ts"] },
});
