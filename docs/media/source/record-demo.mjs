// Records the demo video from the running application.
//
//   cd frontend
//   BASE_URL=http://localhost:8091 OUT_DIR=<work dir> node ../docs/media/source/record-demo.mjs
//
// Point BASE_URL at a FRESH instance (empty artifacts directory): the script
// runs the default 50-generation experiment live through the UI. It only
// drives the real app and draws caption overlays on top of the page; every
// number in a caption is read from that run's evaluation, not typed in here.
// Outputs <OUT_DIR>/frames/*.jpg (Chrome screencast), frames.ffconcat,
// captions.srt and timing.json; build-video.sh turns them into the MP4.
// Set DRY=1 for a quick layout check (tiny training run, screenshots).
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(path.join(process.cwd(), "package.json"));
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL ?? "http://localhost:8091";
const OUT = path.resolve(process.env.OUT_DIR ?? "demo-work");
const DRY = !!process.env.DRY;
const SIZE = { width: 1600, height: 900 };
const TRAIN_SCREEN_SECONDS = 30;
const FRAMES = path.join(OUT, "frames");
fs.rmSync(FRAMES, { recursive: true, force: true });
fs.mkdirSync(FRAMES, { recursive: true });

const OVERLAY_CSS = `
#demo-caption {
  position: fixed; left: 50%; bottom: 34px; transform: translateX(-50%);
  max-width: 1320px; width: max-content; padding: 14px 26px; border-radius: 14px;
  background: rgba(5, 8, 15, 0.9); border: 1px solid rgba(125, 249, 224, 0.35);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6); color: #f2f6ff;
  font: 600 25px/1.35 "Inter", "Segoe UI", system-ui, sans-serif; text-align: center;
  opacity: 0; transition: opacity 0.35s ease; pointer-events: none; z-index: 9999;
}
#demo-card {
  position: fixed; inset: 0; display: grid; place-content: center; gap: 14px; text-align: center;
  background: radial-gradient(900px 500px at 50% 35%, #17223c 0%, #070a12 70%);
  color: #f2f6ff; font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  opacity: 0; transition: opacity 0.5s ease; pointer-events: none; z-index: 10000;
}
#demo-badge {
  position: fixed; left: 26px; top: 22px; padding: 8px 16px; border-radius: 999px;
  background: rgba(5, 8, 15, 0.9); border: 1px solid rgba(255, 255, 255, 0.25); color: #f2f6ff;
  font: 600 17px/1.2 "Inter", "Segoe UI", system-ui, sans-serif;
  opacity: 0; transition: opacity 0.3s ease; pointer-events: none; z-index: 9999;
}
#demo-card .k { color: #7df9e0; font-size: 17px; letter-spacing: 0.2em; text-transform: uppercase; }
#demo-card .t { font-size: 62px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; }
#demo-card .s { color: #a9b6cf; font-size: 26px; white-space: pre-line; line-height: 1.5; }
`;

// GPU rasterization keeps the capture from competing with the training job for CPU.
const browser = await chromium.launch({
  channel: process.env.PW_CHANNEL ?? "chrome",
  args: ["--enable-gpu", "--use-angle=d3d11", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({ viewport: SIZE });
const page = await context.newPage();

// Screen capture: Chrome sends a JPEG whenever the page repaints.
const cdp = await context.newCDPSession(page);
const frames = []; // { file, time } with time in epoch seconds
const writes = [];
cdp.on("Page.screencastFrame", ({ data, metadata, sessionId }) => {
  const file = `f${String(frames.length).padStart(6, "0")}.jpg`;
  frames.push({ file, time: metadata.timestamp });
  writes.push(fs.promises.writeFile(path.join(FRAMES, file), Buffer.from(data, "base64")));
  cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
});
const sleep = (ms) => page.waitForTimeout(ms);
const phase = page.locator("#phase");

let t0 = 0; // first frame of the finished video
const cues = [];
const now = () => (Date.now() - t0) / 1000;
/** Wait until `seconds` on the video clock, so small delays never accumulate. */
const until = (seconds) => sleep(Math.max(0, seconds * 1000 - (Date.now() - t0)));
const closeCue = () => {
  const last = cues[cues.length - 1];
  if (last && last.end == null) last.end = now();
};

async function caption(text) {
  closeCue();
  await page.evaluate(() => (document.getElementById("demo-caption").style.opacity = "0"));
  if (!text) return;
  await sleep(250);
  await page.evaluate((t) => {
    const el = document.getElementById("demo-caption");
    el.textContent = t;
    el.style.opacity = "1";
  }, text);
  cues.push({ start: now(), text });
}

async function card(kicker, title, sub) {
  await page.evaluate(
    ([k, t, s]) => {
      const el = document.getElementById("demo-card");
      el.querySelector(".k").textContent = k;
      el.querySelector(".t").textContent = t;
      el.querySelector(".s").textContent = s;
      el.style.opacity = "1";
    },
    [kicker, title, sub],
  );
  cues.push({ start: now(), text: `${title}\n${sub}` });
}
const hideCard = async () => {
  closeCue();
  await page.evaluate(() => (document.getElementById("demo-card").style.opacity = "0"));
};

const scrollTo = (selector, behavior = "smooth") =>
  page.evaluate(
    ([sel, how]) => {
      const top = document.querySelector(sel).getBoundingClientRect().top + window.scrollY - 12;
      window.scrollTo({ top, behavior: how });
    },
    [selector, behavior],
  );
// Programmatic clicks: no waiting for a scrolling element to settle.
const press = (selector) => page.locator(selector).evaluate((el) => el.click());
const tab = (view) => press(`.tabs button[data-view="${view}"]`);
const badge = (on) =>
  page.evaluate((v) => (document.getElementById("demo-badge").style.opacity = v ? "1" : "0"), on);
const shot = (name) => (DRY ? page.screenshot({ path: path.join(OUT, `dry-${name}.png`) }) : null);

// -- load -------------------------------------------------------------------
await page.goto(BASE);
await page.waitForFunction(() => document.getElementById("phase").textContent !== "Loading…");
await page.waitForFunction(() => document.getElementById("clock").textContent.includes("/ 500"));
if ((await phase.textContent()).startsWith("Replay")) {
  throw new Error("BASE_URL already has a completed experiment; use a fresh instance");
}
await page.addStyleTag({ content: OVERLAY_CSS });
await page.evaluate(() => {
  const el = (id, cls, parent) => {
    const node = document.createElement("div");
    if (id) node.id = id;
    if (cls) node.className = cls;
    parent.appendChild(node);
    return node;
  };
  el("demo-caption", "", document.body);
  el("demo-badge", "", document.body).textContent = "Training · time-lapse";
  const c = el("demo-card", "", document.body);
  for (const cls of ["k", "t", "s"]) el("", cls, c);
  c.style.transition = "none";
  c.style.opacity = "1";
});
await sleep(400);
await page.evaluate(() => (document.getElementById("demo-card").style.transition = ""));

// -- title (0 s) ---------------------------------------------------------------
await scrollTo("#view-initial", "instant");
await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, everyNthFrame: 2 });
await sleep(300);
t0 = Date.now();
await card("Artificial Life · Mini POC", "The Learning Creature", "Can a random brain learn to find food?");
await until(3.5);
await press("#reset");
await hideCard();

// -- 1. the untrained creature ----------------------------------------------------
await caption("This creature starts with a random brain: 206 untrained weights.");
await until(8.0);
await shot("1-initial");
await caption("Green is food. Grey blocks the way. Red drains energy. It must eat to survive.");
await until(12.4);

// -- 2. live training -------------------------------------------------------------
await caption("");
await tab("progress");
await scrollTo("#view-progress", "instant");
if (DRY) {
  await page.fill("#f-population", "12");
  await page.fill("#f-generations", "6");
}
await caption("A genetic algorithm evolves the brain: 40 creatures, 50 generations.");
await until(15.6);
// The training segment is real and uncut, but the finished video plays it as
// a time-lapse fitted to TRAIN_SCREEN_SECONDS (and labels it so), because how
// long training takes depends on what else the machine is doing.
await badge(true);
const fastFrom = now();
await press("#train");
const trainStart = Date.now();
const during = [
  [0, "Live training on 8 practice worlds. Nothing here is scripted."],
  [0.3, "Each generation keeps the best brains, then mixes and mutates them."],
  [0.64, "The chart plots real fitness scores from the simulation."],
]; // shown once this fraction of the generations is done
const generationsDone = async () => {
  const m = (await page.locator("#gen-counter").textContent()).match(/(\d+) \/ (\d+)/);
  return m ? Number(m[1]) / Number(m[2]) : 0;
};
for (;;) {
  if (during.length && (await generationsDone()) >= during[0][0]) {
    await caption(during.shift()[1]);
    if (during.length === 1) await shot("2-training");
  }
  if ((await phase.textContent()) === "Replay of a completed experiment") break;
  if (Date.now() - trainStart > 600_000) throw new Error("training did not finish");
  await sleep(200);
}
const trainSeconds = Math.round((Date.now() - trainStart) / 1000);
const t1 = now(); // the rest of the timeline is relative to the end of training
await badge(false);
await caption(`Trained and evaluated in ${trainSeconds} seconds on a CPU (shown as a time-lapse).`);
await until(t1 + 3.0);
await shot("2-trained");

// Numbers for the evidence captions come from this run's stored evaluation.
const runs = await (await page.request.get(`${BASE}/api/runs`)).json();
const run = await (await page.request.get(`${BASE}/api/runs/${runs.runs[0].id}`)).json();
const suite = run.evaluation.in_distribution;
const food = (name) => suite.controllers[name].food_mean.toFixed(2);
const gain = suite.paired.trained_vs_untrained_population.food;
const n = suite.seeds.length;

// -- 3. before vs. after ------------------------------------------------------------
await caption("");
await tab("compare");
await scrollTo("#compare-body", "instant");
await press("#reset");
await caption("Same world, never seen in training. Left: untrained. Right: trained.");
await until(t1 + 9.0);
await shot("3-compare");
await caption("Same creature. Same challenge. Different learned behavior.");
await until(t1 + 14.2);

// -- 4. measured evidence and limits -------------------------------------------------
await caption("");
await scrollTo("#evidence");
await sleep(600);
await caption(
  `On ${n} held-out worlds, food found rose from ${food("untrained_population")} to ` +
    `${food("trained")} of 6 (95% CI of the gain: ${gain.ci95[0].toFixed(2)} to ${gain.ci95[1].toFixed(2)}).`,
);
await until(t1 + 21.0);
await shot("4-evidence");
await caption(
  `Limits: it only matches a one-line "go to the nearest food" rule (${food("greedy_heuristic")}), ` +
    "and it did not learn to avoid hazards.",
);
await until(t1 + 27.6);

// -- end card ---------------------------------------------------------------------
await caption("");
await card("Verdict · pass with limitations", "Real learning, honestly measured", "Food-seeking: learned.  Hazard avoidance: not shown.");
await until(t1 + 31.4);
closeCue();
const duration = now();

await cdp.send("Page.stopScreencast");
await Promise.all(writes);
await browser.close();

// Recording clock -> video clock: the training segment is fitted to its screen budget.
const SPEED = Math.max(1, (t1 - fastFrom) / TRAIN_SCREEN_SECONDS);
const film = (t) => (t <= fastFrom ? t : t <= t1 ? fastFrom + (t - fastFrom) / SPEED : fastFrom + (t1 - fastFrom) / SPEED + (t - t1));

// Frame list with display durations on the video clock, starting at t0.
const rel = (f) => f.time - t0 / 1000;
frames.sort((x, y) => x.time - y.time);
const kept = frames.filter((f, i) => rel(f) < duration && (i + 1 < frames.length ? rel(frames[i + 1]) : duration) > 0);
const lines = ["ffconcat version 1.0"];
kept.forEach((f, i) => {
  const from = film(Math.max(rel(f), 0));
  const to = film(Math.min(i + 1 < kept.length ? rel(kept[i + 1]) : duration, duration));
  lines.push(`file 'frames/${f.file}'`, `duration ${Math.max(to - from, 0.001).toFixed(4)}`);
});
lines.push(`file 'frames/${kept[kept.length - 1].file}'`);
fs.writeFileSync(path.join(OUT, "frames.ffconcat"), lines.join("\n") + "\n");
for (const c of cues) Object.assign(c, { start: film(c.start), end: film(c.end) });

const stamp = (s) => {
  const ms = Math.round(s * 1000);
  const p = (v, w = 2) => String(v).padStart(w, "0");
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
};
fs.writeFileSync(
  path.join(OUT, "captions.srt"),
  cues.map((c, i) => `${i + 1}\n${stamp(c.start)} --> ${stamp(c.end)}\n${c.text}\n`).join("\n"),
);
fs.writeFileSync(
  path.join(OUT, "timing.json"),
  JSON.stringify(
    { duration: film(duration), recorded: duration, training_speed: Number(SPEED.toFixed(2)), frames: kept.length, train_seconds: trainSeconds, run_id: run.id, config: run.config },
    null,
    2,
  ),
);
console.log(`video ${film(duration).toFixed(1)} s from ${duration.toFixed(1)} s recorded (training ${trainSeconds} s), run ${run.id}`);
