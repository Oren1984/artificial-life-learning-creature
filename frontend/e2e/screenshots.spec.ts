// Captures the README screenshots from the running app. Not part of the test
// suite: run with SCREENSHOTS=1 against a fresh instance (it runs the default
// 50-generation experiment through the UI; if one already exists, only the
// comparison images are refreshed).
import { expect, test } from "@playwright/test";

const out = (name: string) => `../docs/images/${name}.png`;
const stepAtLeast = (n: number) => async (text: string | null) =>
  Number(text?.match(/step (\d+)/)?.[1] ?? 0) >= n;

test.skip(!process.env.SCREENSHOTS, "set SCREENSHOTS=1 to capture documentation images");

test("capture documentation screenshots", async ({ page }, info) => {
  test.setTimeout(600_000);
  const mobile = info.project.name === "mobile";
  await page.goto("/");
  const clock = page.locator("#clock");
  const waitStep = async (n: number) =>
    expect.poll(async () => stepAtLeast(n)(await clock.textContent()), { timeout: 60_000 }).toBe(true);

  await expect(page.locator("#phase")).not.toHaveText("Loading…");
  const trained = (await page.locator("#phase").textContent())!.startsWith("Replay");
  if (!mobile) await page.setViewportSize({ width: 1280, height: 1010 });

  if (!mobile && !trained) {
    await waitStep(150);
    await page.screenshot({ path: out("01-initial-creature") });

    await page.getByRole("button", { name: /Learning progress/ }).click();
    await page.getByRole("button", { name: "Start training" }).click();
    await expect(page.locator("#gen-counter")).toHaveText(/Generation (1[4-9]|2\d) \/ 50/, {
      timeout: 300_000,
    });
    await page.screenshot({ path: out("02-training-live") });
    await expect(page.locator("#phase")).toHaveText("Replay of a completed experiment", {
      timeout: 500_000,
    });
    await page.click("#reset");
    await waitStep(120);
    await page.screenshot({ path: out("03-training-complete") });
  }

  await expect(page.locator("#phase")).toHaveText("Replay of a completed experiment");
  await page.getByRole("button", { name: /Before vs. after/ }).click();
  await page.click("#reset");
  await waitStep(105);
  await page.screenshot({ path: out(mobile ? "06-mobile-before-after" : "04-before-after") });
  if (!mobile) await page.locator("#evidence").screenshot({ path: out("05-measured-evidence") });
});
