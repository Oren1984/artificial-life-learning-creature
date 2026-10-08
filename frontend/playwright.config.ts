import { defineConfig } from "@playwright/test";

// Runs against the already-started app (docker compose up, or native uvicorn).
// Uses the locally installed Chrome so no browser download is needed.
export default defineConfig({
  testDir: "e2e",
  workers: 1, // the backend runs one training job at a time
  timeout: 180_000,
  reporter: [["list"]],
  use: {
    baseURL: process.env.BASE_URL ?? "http://localhost:8080",
    channel: process.env.PW_CHANNEL ?? "chrome",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1280, height: 860 } } },
    {
      name: "mobile",
      use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
    },
  ],
});
