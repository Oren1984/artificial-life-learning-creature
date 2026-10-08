import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp, esc, type App } from "../src/app";
import type { Run } from "../src/types";
import { fakeApi, gen, makeRun } from "./fixtures";

const html = readFileSync(resolve(__dirname, "../index.html"), "utf8");
const css = readFileSync(resolve(__dirname, "../src/style.css"), "utf8");
const $ = (id: string) => document.getElementById(id)!;
const settle = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};

let app: App;

async function mount(run: Run | null) {
  const fake = fakeApi(run);
  app = createApp(document, fake.api);
  await app.ready;
  return fake;
}

beforeEach(() => {
  vi.useFakeTimers();
  document.body.innerHTML = html.slice(html.indexOf("<body>") + 6, html.indexOf("<script"));
  // jsdom has no canvas; rendering is exercised in the browser E2E tests.
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;
});

afterEach(() => {
  app?.destroy();
  vi.useRealTimers();
});

describe("initialization", () => {
  it("shows the untrained creature when no experiment exists", async () => {
    const { api } = await mount(null);
    expect($("phase").textContent).toBe("No experiment yet");
    expect($("view-initial").hidden).toBe(false);
    expect($("view-progress").hidden).toBe(true);
    expect(api.replay).toHaveBeenCalledWith(null, 9000, "standard", [{ kind: "baseline" }]);
    expect($("stats-initial").textContent).toContain("Food");
    expect(($("f-generations") as HTMLInputElement).value).toBe("50");
    expect($("run-select").textContent).toContain("None yet");
  });

  it("loads a completed experiment as a replay, not as live training", async () => {
    await mount(makeRun("completed", [gen(0, 5, 1), gen(1, 30, 9)]));
    expect($("phase").textContent).toBe("Replay of a completed experiment");
    expect($("phase").dataset.phase).toBe("completed");
    expect($("gen-counter").textContent).toBe("Generation 1 / 50");
  });

  it("reports an unreachable backend instead of rendering stale data", async () => {
    const fake = fakeApi(null);
    fake.api.config = vi.fn(async () => {
      throw new Error("request failed (502)");
    });
    app = createApp(document, fake.api);
    await app.ready;
    expect($("error").hidden).toBe(false);
    expect($("error").textContent).toBe("request failed (502)");
    expect($("phase").textContent).toBe("Backend unavailable");
  });
});

describe("training progress", () => {
  it("starts training, polls real metrics and finishes", async () => {
    const { api, state } = await mount(null);
    (document.querySelector('[data-view="progress"]') as HTMLElement).click();
    expect($("progress-title").textContent).toContain("No generations yet");

    $("train-form").dispatchEvent(new Event("submit", { cancelable: true }));
    await settle();
    expect(api.startTraining).toHaveBeenCalledWith({ population: 40, generations: 50, seed: 0 });
    expect($("phase").textContent).toBe("Live training");
    expect(($("train") as HTMLButtonElement).disabled).toBe(true);
    expect($("gen-counter").textContent).toBe("Generation 0 / 50");

    state.run = makeRun("running", [gen(0, 5, 1), gen(1, 12, 3), gen(2, 25.5, 8)]);
    await vi.advanceTimersByTimeAsync(600);
    expect($("gen-counter").textContent).toBe("Generation 2 / 50");
    expect($("tiles").textContent).toContain("25.5");
    expect($("gen-table").querySelectorAll("tr").length).toBe(4);

    state.run = makeRun("completed", [gen(0, 5, 1), gen(1, 12, 3), gen(2, 25.5, 8), gen(3, 40, 20)]);
    await vi.advanceTimersByTimeAsync(600);
    expect($("phase").textContent).toBe("Replay of a completed experiment");
    expect(($("train") as HTMLButtonElement).disabled).toBe(false);
    const calls = (api.run as ReturnType<typeof vi.fn>).mock.calls.length;
    await vi.advanceTimersByTimeAsync(3000);
    expect((api.run as ReturnType<typeof vi.fn>).mock.calls.length).toBe(calls); // polling stopped
  });

  it("surfaces a rejected training request", async () => {
    const { api } = await mount(null);
    api.startTraining = vi.fn(async () => {
      throw new Error("a training run is already in progress");
    });
    $("train-form").dispatchEvent(new Event("submit", { cancelable: true }));
    await settle();
    expect($("error").textContent).toBe("a training run is already in progress");
    expect($("phase").textContent).toBe("No experiment yet");
  });

  it("reports a failed run as failed", async () => {
    const { state } = await mount(makeRun("running", [gen(0, 5, 1)]));
    state.run = makeRun("failed", [gen(0, 5, 1)]);
    await vi.advanceTimersByTimeAsync(600);
    expect($("phase").textContent).toBe("Training failed");
    expect($("error").hidden).toBe(false);
    (document.querySelector('[data-view="compare"]') as HTMLElement).click();
    expect($("compare-empty").hidden).toBe(false);
    expect($("evidence").hidden).toBe(true);
  });

  it("replays the generation chosen with the slider", async () => {
    const { api } = await mount(makeRun("completed", [gen(0, 5, 1), gen(1, 30, 9), gen(2, 44, 20)]));
    (document.querySelector('[data-view="progress"]') as HTMLElement).click();
    await settle();
    const slider = $("gen-slider") as HTMLInputElement;
    expect(slider.max).toBe("2");
    slider.value = "0";
    slider.dispatchEvent(new Event("input"));
    await settle();
    expect(api.replay).toHaveBeenLastCalledWith(expect.any(String), 9000, "standard", [
      { kind: "generation", generation: 0 },
    ]);
    expect($("progress-title").textContent).toBe("Best creature of generation 0");
  });
});

describe("before/after comparison", () => {
  it("requests both controllers for one identical scenario", async () => {
    const { api } = await mount(makeRun("completed", [gen(0, 5, 1), gen(1, 30, 9)]));
    (document.querySelector('[data-view="compare"]') as HTMLElement).click();
    await settle();
    const calls = (api.replay as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls[calls.length - 1]).toEqual([
      "20261008-120000-abcdef",
      9000,
      "standard",
      [{ kind: "baseline" }, { kind: "best" }],
    ]);
    expect($("compare-body").hidden).toBe(false);
    expect($("stats-before").textContent).toContain("Food");
    expect($("stats-after").textContent).toContain("Food");
    expect($("verdict").dataset.outcome).toBe("improved");
    expect($("eval-tables").textContent).toContain("Trained creature");
    expect($("eval-tables").textContent).toContain("never trained on");
  });

  it("shows a weak result as weak", async () => {
    const run = makeRun("completed", [gen(0, 5, 1)]);
    run.evaluation!.verdict.outcome = "no_clear_evidence";
    await mount(run);
    (document.querySelector('[data-view="compare"]') as HTMLElement).click();
    await settle();
    expect($("verdict").textContent).toContain("No clear evidence of improvement");
  });

  it("offers nothing to compare before training has completed", async () => {
    const { api } = await mount(null);
    (document.querySelector('[data-view="compare"]') as HTMLElement).click();
    await settle();
    expect($("compare-empty").hidden).toBe(false);
    expect($("compare-body").hidden).toBe(true);
    expect((api.replay as ReturnType<typeof vi.fn>).mock.calls.length).toBe(1); // only the initial view
  });
});

describe("controls", () => {
  it("pauses, resumes and resets the shared clock", async () => {
    await mount(null);
    expect($("play").textContent).toBe("Pause");
    $("play").click();
    expect($("play").textContent).toBe("Play");
    $("reset").click();
    expect($("play").textContent).toBe("Pause");
    expect($("clock").textContent).toBe("step 0 / 500");
  });

  it("reloads the replay when the scenario changes", async () => {
    const { api } = await mount(null);
    const variant = $("variant") as HTMLSelectElement;
    variant.value = "dense_hazards";
    variant.dispatchEvent(new Event("change"));
    await settle();
    expect(api.replay).toHaveBeenLastCalledWith(null, 9100, "dense_hazards", [{ kind: "baseline" }]);
    expect($("seed").textContent).toContain("9101");
  });

  it("shows a replay error and recovers", async () => {
    const { api } = await mount(null);
    const ok = api.replay;
    api.replay = vi.fn(async () => {
      throw new Error("internal error");
    });
    const seed = $("seed") as HTMLSelectElement;
    seed.value = "9001";
    seed.dispatchEvent(new Event("change"));
    await settle();
    expect($("error").textContent).toBe("internal error");
    api.replay = ok;
    seed.value = "9000";
    seed.dispatchEvent(new Event("change"));
    await settle();
    expect($("stats-initial").textContent).toContain("Food");
  });
});

describe("markup and layout basics", () => {
  it("escapes text placed into markup", () => {
    expect(esc('<img src=x onerror="1">&')).toBe("&lt;img src=x onerror=&quot;1&quot;&gt;&amp;");
  });

  it("declares a mobile viewport and collapses to one column on small screens", () => {
    expect(html).toContain('name="viewport" content="width=device-width, initial-scale=1"');
    const mobile = css.slice(css.indexOf("@media (max-width: 860px)"));
    expect(mobile).toContain("grid-template-columns: minmax(0, 1fr)");
  });
});
