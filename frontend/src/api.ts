import type { Api } from "./types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* keep the generic message */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const httpApi: Api = {
  config: () => request("/api/config"),
  runs: async () => (await request<{ runs: never[] }>("/api/runs")).runs,
  run: (id) => request(`/api/runs/${id}`),
  startTraining: async (cfg) =>
    (await request<{ run_id: string }>("/api/training", cfg)).run_id,
  replay: (runId, seed, variant, controllers) =>
    request("/api/replay", { run_id: runId, seed, variant, controllers }),
};
