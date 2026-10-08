import { expect, test, type Page } from "@playwright/test";

const problems = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const found: string[] = [];
  problems.set(page, found);
  page.on("pageerror", (e) => found.push(`pageerror: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && found.push(`console: ${m.text()}`));
  page.on("requestfailed", (r) => found.push(`requestfailed: ${r.url()}`));
  page.on("response", (r) => r.status() >= 400 && found.push(`${r.status()}: ${r.url()}`));
  await page.goto("/");
  await expect(page.locator("#phase")).not.toHaveText("Loading…");
  await expect(page.locator("#clock")).toContainText("/ 500"); // first replay has arrived
});

test.afterEach(async ({ page }) => {
  expect(problems.get(page)).toEqual([]);
});

/** Number of distinct colours on a canvas: a blank canvas has one. */
async function colours(page: Page, id: string): Promise<number> {
  return page.evaluate((canvasId) => {
    const c = document.getElementById(canvasId) as HTMLCanvasElement;
    const data = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    const seen = new Set<number>();
    for (let i = 0; i < data.length; i += 4 * 97) {
      seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
    }
    return seen.size;
  }, id);
}

/** Wait for the replay response that answers a specific request. */
function replayFor(page: Page, match: (body: any) => boolean) {
  return page.waitForResponse(
    (r) => r.url().endsWith("/api/replay") && match(r.request().postDataJSON()),
  );
}

async function train(page: Page) {
  await page.getByRole("button", { name: /Learning progress/ }).click();
  await page.fill("#f-population", "12");
  await page.fill("#f-generations", "6");
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page.locator("#phase")).toHaveText("Live training");
  await expect(page.locator("#train")).toBeDisabled();
  await expect(page.locator("#phase")).toHaveText("Replay of a completed experiment", {
    timeout: 120_000,
  });
  await expect(page.locator("#gen-counter")).toHaveText("Generation 6 / 6");
}

test("loads and renders the untrained creature", async ({ page }) => {
  await expect(page).toHaveTitle("The Learning Creature");
  await expect(page.locator("#canvas-initial")).toBeVisible();
  await expect(page.locator("#stats-initial")).toContainText("Food");
  expect(await colours(page, "canvas-initial")).toBeGreaterThan(20);
  await expect(page.locator("#clock")).not.toHaveText("step 0 / 500");
});

test("pause, resume and reset control the replay clock", async ({ page }) => {
  await page.selectOption("#speed", "0.5");
  await page.click("#reset");
  await page.click("#play");
  await expect(page.locator("#play")).toHaveText("Play");
  const frozen = await page.locator("#clock").textContent();
  await page.waitForTimeout(400);
  await expect(page.locator("#clock")).toHaveText(frozen!);
  await page.click("#play");
  await expect(page.locator("#clock")).not.toHaveText(frozen!);
  await page.click("#reset");
  const step = Number((await page.locator("#clock").textContent())!.match(/step (\d+)/)![1]);
  expect(step).toBeLessThan(30);
});

test("trains live, then compares before and after on the same scenario", async ({ page }) => {
  await train(page);
  expect(await colours(page, "chart")).toBeGreaterThan(3);
  await expect(page.locator("#gen-table tr")).toHaveCount(8);
  await expect(page.locator("#progress-title")).toHaveText("Best creature of generation 6");
  expect(await colours(page, "canvas-progress")).toBeGreaterThan(20);

  await page.locator("#gen-slider").fill("0");
  await expect(page.locator("#progress-title")).toHaveText("Best creature of generation 0");

  const replayed = replayFor(page, (b) => b.controllers.length === 2);
  await page.getByRole("button", { name: /Before vs. after/ }).click();
  const body = await (await replayed).json();
  expect(body.replays.map((r: { controller: string }) => r.controller)).toEqual(["baseline", "best"]);
  expect(body.replays[0].x[0]).toBe(body.replays[1].x[0]); // same start, same layout
  expect(await colours(page, "canvas-before")).toBeGreaterThan(20);
  expect(await colours(page, "canvas-after")).toBeGreaterThan(20);
  await expect(page.locator("#verdict")).toHaveAttribute(
    "data-outcome",
    /improved|no_clear_evidence|regressed/,
  );
  await expect(page.locator("#eval-tables table")).toHaveCount(4);

  // The outcome shown at the end is the one the backend measured.
  await page.selectOption("#speed", "4");
  await expect(page.locator("#stats-before .outcome")).not.toHaveText("Episode in progress…", {
    timeout: 30_000,
  });
  const s = body.replays[0].summary;
  const expected = s.success ? "Collected all food" : s.dead ? "Ran out of energy" : "Time ran out";
  await expect(page.locator("#stats-before .outcome")).toContainText(expected);
});

test("switching scenario loads a different layout", async ({ page }) => {
  const replayed = replayFor(page, (b) => b.variant === "dense_hazards");
  await page.selectOption("#variant", "dense_hazards");
  const body = await (await replayed).json();
  expect(body.scenario.variant).toBe("dense_hazards");
  expect(body.scenario.hazards).toHaveLength(6);
  await expect(page.locator("#seed")).toHaveValue("9100");
  const next = replayFor(page, (b) => b.seed === 9105);
  await page.selectOption("#seed", "9105");
  expect((await (await next).json()).scenario.seed).toBe(9105);
});

test("a completed experiment survives a reload", async ({ page }) => {
  await train(page);
  const run = await page.locator("#run-select").inputValue();
  await page.reload();
  await expect(page.locator("#phase")).toHaveText("Replay of a completed experiment");
  await expect(page.locator("#run-select")).toHaveValue(run);
  await page.getByRole("button", { name: /Learning progress/ }).click();
  await expect(page.locator("#gen-counter")).toHaveText("Generation 6 / 6");
});

test("layout fits the viewport without horizontal scrolling", async ({ page }) => {
  for (const view of [/Initial creature/, /Learning progress/, /Before vs. after/]) {
    await page.getByRole("button", { name: view }).click();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  }
  await page.getByRole("button", { name: /Initial creature/ }).click();
  const box = await page.locator("#canvas-initial").boundingBox();
  expect(box!.width).toBeGreaterThan(250);
  expect(box!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
});
