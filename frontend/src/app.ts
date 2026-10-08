import { FitnessChart } from "./chart";
import {
  PHASE_LABEL,
  SUITE_HEADERS,
  VARIANT_LABEL,
  VERDICT_TEXT,
  advanceClock,
  frameAt,
  outcomeText,
  phaseOf,
  replayEnd,
  suiteRows,
} from "./logic";
import { BASELINE_STYLE, TRAINED_STYLE, drawPane, type PaneStyle } from "./render";
import type { Api, AppConfig, ControllerRef, Replay, ReplayResponse, Run, Suite } from "./types";

type View = "initial" | "progress" | "compare";
const POLL_MS = 500;

const ESCAPES: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
/** Escape text before it is placed into markup. */
export const esc = (value: unknown): string => String(value).replace(/[&<>"]/g, (c) => ESCAPES[c]);

export interface App {
  ready: Promise<void>;
  destroy(): void;
}

export function createApp(doc: Document, api: Api): App {
  const $ = <T extends HTMLElement = HTMLElement>(id: string) => doc.getElementById(id) as T;
  const win = doc.defaultView!;

  let config: AppConfig;
  let run: Run | null = null;
  let view: View = "initial";
  let replay: ReplayResponse | null = null;
  let replayKey = "";
  let generation = 0;
  let follow = true; // during live training, keep showing the newest generation
  let t = 0;
  let playing = true;
  let lastTick = 0;
  let frameHandle = 0;
  let pollHandle = 0;
  let alive = true;

  const chart = new FitnessChart($<HTMLCanvasElement>("chart"), $("chart-tip"), (g) => {
    follow = false;
    setGeneration(g);
  });

  function showError(message: string | null): void {
    const box = $("error");
    box.hidden = !message;
    box.textContent = message ?? "";
  }

  async function guard<T>(work: () => Promise<T>): Promise<T | undefined> {
    try {
      const result = await work();
      return result;
    } catch (err) {
      showError(err instanceof Error ? err.message : "Something went wrong");
      return undefined;
    }
  }

  // -- replay loading ------------------------------------------------------
  function wanted(): ControllerRef[] | null {
    if (view === "initial") return [{ kind: "baseline" }];
    if (view === "progress") {
      return run && run.generations.length ? [{ kind: "generation", generation }] : null;
    }
    return run && run.state === "completed" ? [{ kind: "baseline" }, { kind: "best" }] : null;
  }

  async function loadReplay(): Promise<void> {
    const controllers = wanted();
    const seed = Number($<HTMLSelectElement>("seed").value);
    const variant = $<HTMLSelectElement>("variant").value;
    const key = JSON.stringify([view, run?.id ?? null, seed, variant, controllers]);
    if (key === replayKey) return;
    replayKey = key;
    replay = null;
    t = 0;
    if (!controllers) return renderStatic();
    const data = await guard(() => api.replay(run?.id ?? null, seed, variant, controllers));
    if (key !== replayKey || !data) return; // superseded or failed
    replay = data;
    t = 0;
    playing = true;
    renderStatic();
  }

  // -- rendering -----------------------------------------------------------
  function panes(): [string, string, PaneStyle][] {
    if (view === "initial") return [["canvas-initial", "stats-initial", BASELINE_STYLE]];
    if (view === "progress") return [["canvas-progress", "stats-progress", TRAINED_STYLE]];
    return [
      ["canvas-before", "stats-before", BASELINE_STYLE],
      ["canvas-after", "stats-after", TRAINED_STYLE],
    ];
  }

  function statsHtml(r: Replay, time: number): string {
    const f = frameAt(r, time);
    let collisions = 0;
    let hazard = 0;
    for (let i = 1; i <= f.step; i++) {
      if (r.contact[i] && !r.contact[i - 1]) collisions++;
      if (r.in_hazard[i]) hazard++;
    }
    const food = f.foodLeft.filter((left) => !left).length;
    const cell = (value: string | number, label: string) =>
      `<div><b>${esc(value)}</b><small>${esc(label)}</small></div>`;
    return (
      cell(`${food}/${f.foodLeft.length}`, "Food") +
      cell(`${Math.round(f.energy * 100)}%`, "Energy") +
      cell(collisions, "Collisions") +
      cell(hazard, "Hazard steps") +
      `<div class="outcome">${esc(f.ended ? outcomeText(r) : "Episode in progress…")}</div>`
    );
  }

  function drawFrame(now: number): void {
    if (!replay) return;
    const radii = { creature: replay.world.creature_radius, food: replay.world.food_radius };
    const trails = $<HTMLInputElement>("trails").checked;
    panes().forEach(([canvasId, statsId, style], i) => {
      const r = replay!.replays[i];
      if (!r) return;
      drawPane($<HTMLCanvasElement>(canvasId), replay!.scenario, r, t, radii, style, trails, now);
      $(statsId).innerHTML = statsHtml(r, t);
    });
    $("clock").textContent = `step ${Math.floor(t)} / ${replay.world.max_steps}`;
  }

  function tick(now: number): void {
    if (!alive) return;
    const elapsed = lastTick ? Math.min(100, now - lastTick) : 0;
    lastTick = now;
    if (replay) {
      const speed = Number($<HTMLSelectElement>("speed").value);
      t = advanceClock(t, elapsed, speed, playing, replayEnd(replay.replays));
      drawFrame(now);
    }
    frameHandle = win.requestAnimationFrame(tick);
  }

  function table(headers: string[], rows: string[][], highlight?: string): string {
    const head = `<tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr>`;
    const body = rows
      .map(
        (r) =>
          `<tr class="${r[0] === highlight ? "hl" : ""}">${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`,
      )
      .join("");
    return `<div class="scroll"><table>${head}${body}</table></div>`;
  }

  function suiteHtml(title: string, suite: Suite): string {
    const p = suite.paired.trained_vs_untrained_population.food;
    const note =
      `Trained − untrained food: ${p.mean >= 0 ? "+" : ""}${p.mean.toFixed(2)} ` +
      `(95% CI ${p.ci95[0].toFixed(2)} to ${p.ci95[1].toFixed(2)}, ${suite.seeds.length} layouts)`;
    return (
      `<h3>${esc(title)}</h3>` +
      table(SUITE_HEADERS, suiteRows(suite), "Trained creature") +
      `<p class="muted">${esc(note)}</p>`
    );
  }

  /** Everything that only changes when data changes (not every animation frame). */
  function renderStatic(): void {
    const phase = phaseOf(run);
    const pill = $("phase");
    pill.dataset.phase = phase;
    pill.textContent = PHASE_LABEL[phase];

    for (const v of ["initial", "progress", "compare"] as View[]) {
      $(`view-${v}`).hidden = v !== view;
      doc.querySelector(`[data-view="${v}"]`)!.classList.toggle("active", v === view);
    }

    const gens = run?.generations ?? [];
    const total = run?.config.generations ?? config.defaults.generations;
    const latest = gens.length ? gens[gens.length - 1] : null;
    $("gen-counter").textContent = `Generation ${latest ? latest.generation : "–"} / ${total}`;
    chart.update(gens, total, gens.length ? generation : null);
    const tile = (value: string, label: string) =>
      `<div><b>${esc(value)}</b><small>${esc(label)}</small></div>`;
    $("tiles").innerHTML = latest
      ? tile(latest.best_fitness.toFixed(1), "Best fitness") +
        tile(`${latest.best_food.toFixed(2)}/6`, "Best: food") +
        tile(latest.best_collisions.toFixed(2), "Best: collisions") +
        tile(latest.best_survival.toFixed(0), "Best: survival steps")
      : tile("–", "Best fitness") + tile("–", "Best: food") + tile("–", "Best: collisions") + tile("–", "Best: survival steps");
    $("gen-table").innerHTML =
      "<tr><th>Gen</th><th>Best fitness</th><th>Avg fitness</th><th>Best food</th><th>Avg food</th><th>Best collisions</th></tr>" +
      gens
        .map(
          (g) =>
            `<tr><td>${g.generation}</td><td>${g.best_fitness.toFixed(2)}</td><td>${g.mean_fitness.toFixed(2)}</td>` +
            `<td>${g.best_food.toFixed(2)}</td><td>${g.mean_food.toFixed(2)}</td><td>${g.best_collisions.toFixed(2)}</td></tr>`,
        )
        .join("");

    const slider = $<HTMLInputElement>("gen-slider");
    slider.max = String(Math.max(0, gens.length - 1));
    slider.value = String(generation);
    slider.disabled = gens.length === 0;
    $("progress-title").textContent = gens.length
      ? `Best creature of generation ${generation}`
      : "No generations yet. Start training.";
    $<HTMLButtonElement>("train").disabled = phase === "live";
    $("train").textContent = phase === "live" ? "Training…" : "Start training";

    const comparable = !!run && run.state === "completed";
    $("compare-empty").hidden = comparable;
    $("compare-body").hidden = !comparable;
    const ev = comparable ? run!.evaluation : null;
    $("evidence").hidden = !ev;
    if (ev) {
      const verdict = $("verdict");
      verdict.dataset.outcome = ev.verdict.outcome;
      verdict.textContent = VERDICT_TEXT[ev.verdict.outcome];
      $("eval-tables").innerHTML =
        suiteHtml("Held-out layouts from the training distribution", ev.in_distribution) +
        Object.entries(ev.unseen)
          .map(([name, suite]) => suiteHtml(`${VARIANT_LABEL[name]} (never trained on)`, suite))
          .join("");
    }

    $("play").textContent = playing ? "Pause" : "Play";
    if (!replay) {
      $("clock").textContent = "step 0";
      for (const [, statsId] of panes()) $(statsId).innerHTML = "";
    } else {
      drawFrame(win.performance.now());
    }
  }

  // -- state changes -------------------------------------------------------
  function setGeneration(g: number): void {
    generation = g;
    renderStatic();
    void loadReplay();
  }

  async function selectRun(id: string | null): Promise<void> {
    win.clearTimeout(pollHandle);
    run = id ? ((await guard(() => api.run(id))) ?? null) : null;
    follow = true;
    generation = run ? Math.max(0, run.generations.length - 1) : 0;
    renderStatic();
    await loadReplay();
    if (run?.state === "running") pollHandle = win.setTimeout(poll, POLL_MS);
  }

  async function poll(): Promise<void> {
    if (!alive || !run) return;
    const id = run.id;
    const fresh = await guard(() => api.run(id));
    if (!alive || !fresh || run?.id !== id) return;
    run = fresh;
    const ended = !replay || t >= replayEnd(replay.replays);
    if (follow && fresh.generations.length && (ended || fresh.state !== "running")) {
      generation = fresh.generations.length - 1;
    }
    renderStatic();
    void loadReplay();
    if (fresh.state === "running") {
      pollHandle = win.setTimeout(poll, POLL_MS);
    } else {
      await refreshRuns(id);
      if (fresh.state === "failed") showError("Training failed. See the server log for details.");
    }
  }

  async function refreshRuns(selected: string | null): Promise<void> {
    const runs = (await guard(() => api.runs())) ?? [];
    const select = $<HTMLSelectElement>("run-select");
    select.innerHTML = runs.length
      ? runs
          .map((r) => {
            const when = r.created.slice(0, 16).replace("T", " ");
            return `<option value="${esc(r.id)}">${esc(`${when} · seed ${r.config.seed} · ${r.state}`)}</option>`;
          })
          .join("")
      : `<option value="">None yet (untrained creature)</option>`;
    if (selected) select.value = selected;
  }

  function wire(): void {
    doc.querySelectorAll<HTMLElement>("[data-view]").forEach((button) =>
      button.addEventListener("click", () => {
        view = button.dataset.view as View;
        renderStatic();
        void loadReplay();
      }),
    );
    $("run-select").addEventListener("change", (e) => {
      void selectRun((e.target as HTMLSelectElement).value || null);
    });
    for (const id of ["variant", "seed"]) {
      $(id).addEventListener("change", () => {
        if (id === "variant") fillSeeds();
        void loadReplay();
      });
    }
    $("play").addEventListener("click", () => {
      if (replay && t >= replayEnd(replay.replays)) {
        t = 0; // finished: play again from the start
        playing = true;
      } else {
        playing = !playing;
      }
      renderStatic();
    });
    $("reset").addEventListener("click", () => {
      t = 0;
      playing = true;
      renderStatic();
    });
    $("gen-slider").addEventListener("input", (e) => {
      follow = false;
      setGeneration(Number((e.target as HTMLInputElement).value));
    });
    $("train-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      showError(null);
      const num = (id: string) => Number($<HTMLInputElement>(id).value);
      const id = await guard(() =>
        api.startTraining({
          population: num("f-population"),
          generations: num("f-generations"),
          seed: num("f-seed"),
        }),
      );
      if (!id) return;
      await refreshRuns(id);
      await selectRun(id);
    });
  }

  function fillSeeds(): void {
    const variant = $<HTMLSelectElement>("variant").value;
    const seeds = variant === "standard" ? config.held_out_seeds : config.unseen_seeds;
    $("seed").innerHTML = seeds.map((s) => `<option value="${s}">${s}</option>`).join("");
  }

  async function init(): Promise<void> {
    const loaded = await guard(() => api.config());
    if (!loaded) {
      $("phase").textContent = "Backend unavailable";
      return;
    }
    config = loaded;
    $("variant").innerHTML = config.variants
      .map((v) => `<option value="${esc(v)}">${esc(VARIANT_LABEL[v] ?? v)}</option>`)
      .join("");
    fillSeeds();
    $<HTMLInputElement>("f-population").value = String(config.defaults.population);
    $<HTMLInputElement>("f-generations").value = String(config.defaults.generations);
    $<HTMLInputElement>("f-seed").value = String(config.defaults.seed);
    wire();
    const runs = (await guard(() => api.runs())) ?? [];
    const first = runs.find((r) => r.state === "running") ?? runs.find((r) => r.state === "completed");
    await refreshRuns(first?.id ?? null);
    await selectRun(first?.id ?? null);
    win.addEventListener("resize", () => renderStatic());
    frameHandle = win.requestAnimationFrame(tick);
  }

  return {
    ready: init(),
    destroy() {
      alive = false;
      win.cancelAnimationFrame(frameHandle);
      win.clearTimeout(pollHandle);
    },
  };
}
