import { describe, expect, it } from "vitest";
import {
  advanceClock,
  chartSeries,
  frameAt,
  niceTicks,
  outcomeText,
  phaseOf,
  replayEnd,
  scale,
  suiteRows,
} from "../src/logic";
import { evaluation, gen, makeReplay, makeRun } from "./fixtures";

describe("training phase", () => {
  it("distinguishes live training from a completed experiment", () => {
    expect(phaseOf(null)).toBe("empty");
    expect(phaseOf(makeRun("running", []))).toBe("live");
    expect(phaseOf(makeRun("completed", []))).toBe("completed");
    expect(phaseOf(makeRun("failed", []))).toBe("failed");
    expect(phaseOf(makeRun("interrupted", []))).toBe("interrupted");
  });
});

describe("chart data mapping", () => {
  it("maps generation records to series without altering values", () => {
    const s = chartSeries([gen(0, 11.4, 2.07), gen(1, 20, 5), gen(2, 19.5, 7)]);
    expect(s.x).toEqual([0, 1, 2]);
    expect(s.best).toEqual([11.4, 20, 19.5]); // a drop is shown as a drop
    expect(s.mean).toEqual([2.07, 5, 7]);
    expect(chartSeries([])).toEqual({ x: [], best: [], mean: [] });
  });

  it("scales linearly and survives an empty domain", () => {
    const sy = scale([0, 100], [200, 0]);
    expect(sy(0)).toBe(200);
    expect(sy(50)).toBe(100);
    expect(scale([5, 5], [0, 10])(5)).toBe(0);
  });

  it("produces round ticks that cover the data", () => {
    const ticks = niceTicks(-3, 78);
    expect(ticks[0]).toBeLessThanOrEqual(-3);
    expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(78);
    expect(new Set(ticks.map((t, i) => (i ? t - ticks[i - 1] : 0)).slice(1)).size).toBe(1);
    expect(niceTicks(0, 0).length).toBeGreaterThan(1);
  });
});

describe("replay clock", () => {
  const replay = makeReplay("baseline", 1, 4);

  it("interpolates between recorded simulation steps", () => {
    const f = frameAt(replay, 1.5);
    expect(f.x).toBeCloseTo(0.25);
    expect(f.step).toBe(1);
    expect(f.ended).toBe(false);
    expect(f.foodLeft).toEqual([true, true]);
  });

  it("freezes on the final recorded frame", () => {
    const f = frameAt(replay, 999);
    expect(f.x).toBeCloseTo(0.5);
    expect(f.step).toBe(4);
    expect(f.ended).toBe(true);
    expect(f.foodLeft).toEqual([false, true]);
    expect(frameAt(replay, -5).x).toBeCloseTo(0.1);
  });

  it("advances by speed, stops when paused and clamps at the end", () => {
    expect(advanceClock(0, 1000, 1, true, 500)).toBe(60);
    expect(advanceClock(0, 1000, 4, true, 500)).toBe(240);
    expect(advanceClock(10, 1000, 1, false, 500)).toBe(10);
    expect(advanceClock(490, 1000, 1, true, 500)).toBe(500);
  });

  it("uses one shared end for replays of different length", () => {
    expect(replayEnd([makeReplay("a", 0, 4), makeReplay("b", 1, 9)])).toBe(9);
    expect(replayEnd([])).toBe(0);
  });

  it("describes the outcome that was actually observed", () => {
    expect(outcomeText(replay)).toBe("Ran out of energy at step 4");
    const won = makeReplay("best", 6);
    won.summary = { ...won.summary, success: true, dead: false, steps: 211 };
    expect(outcomeText(won)).toBe("Collected all food in 211 steps");
    const slow = makeReplay("best", 3);
    slow.summary = { ...slow.summary, dead: false };
    expect(outcomeText(slow)).toBe("Time ran out with 3 food collected");
  });
});

describe("evaluation table", () => {
  it("lists every controller with its measured values", () => {
    const rows = suiteRows(evaluation("improved").in_distribution);
    expect(rows.map((r) => r[0])).toEqual([
      "Untrained (10 random networks)",
      "Untrained baseline creature",
      "Random actions",
      "Trained creature",
      "Hand-written greedy heuristic",
    ]);
    expect(rows[3][1]).toBe("5.50 ± 0.50");
    expect(rows[3][2]).toBe("92%");
  });
});
