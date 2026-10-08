// Pure presentation logic. Nothing here simulates the world: every position,
// energy value and outcome is read from data produced by the backend engine.
import type { GenerationRecord, Replay, Run, Suite } from "./types";

export type Phase = "empty" | "live" | "completed" | "failed" | "interrupted";

export function phaseOf(run: Run | null): Phase {
  if (!run) return "empty";
  if (run.state === "running") return "live";
  return run.state;
}

export const PHASE_LABEL: Record<Phase, string> = {
  empty: "No experiment yet",
  live: "Live training",
  completed: "Replay of a completed experiment",
  failed: "Training failed",
  interrupted: "Training was interrupted",
};

export interface ChartSeries {
  x: number[];
  best: number[];
  mean: number[];
}

export function chartSeries(generations: GenerationRecord[]): ChartSeries {
  return {
    x: generations.map((g) => g.generation),
    best: generations.map((g) => g.best_fitness),
    mean: generations.map((g) => g.mean_fitness),
  };
}

export function scale(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (v: number) => range[0] + ((v - domain[0]) / span) * (range[1] - range[0]);
}

/** Round-number axis ticks covering [min, max]. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (max <= min) max = min + 1;
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
  const ticks: number[] = [];
  for (let v = Math.floor(min / step) * step; v <= max + step * 0.999; v += step) {
    ticks.push(Math.round(v / step) * step);
  }
  return ticks;
}

export interface Frame {
  x: number;
  y: number;
  energy: number;
  foodLeft: boolean[];
  contact: boolean;
  inHazard: boolean;
  step: number;
  ended: boolean;
}

/** State of a replay at shared clock time `t` (in simulation steps). */
export function frameAt(replay: Replay, t: number): Frame {
  const last = replay.x.length - 1;
  const clamped = Math.max(0, Math.min(t, last));
  const i = Math.floor(clamped);
  const j = Math.min(i + 1, last);
  const f = clamped - i;
  return {
    x: replay.x[i] + (replay.x[j] - replay.x[i]) * f,
    y: replay.y[i] + (replay.y[j] - replay.y[i]) * f,
    energy: replay.energy[i] + (replay.energy[j] - replay.energy[i]) * f,
    foodLeft: replay.food_left[i],
    contact: replay.contact[i],
    inHazard: replay.in_hazard[i],
    step: i,
    ended: t >= last,
  };
}

export const STEPS_PER_SECOND = 60;

export function advanceClock(
  t: number,
  elapsedMs: number,
  speed: number,
  playing: boolean,
  end: number,
): number {
  if (!playing) return t;
  return Math.min(end, t + (elapsedMs / 1000) * STEPS_PER_SECOND * speed);
}

export function replayEnd(replays: Replay[]): number {
  return Math.max(0, ...replays.map((r) => r.x.length - 1));
}

export function outcomeText(replay: Replay): string {
  const s = replay.summary;
  if (s.success) return `Collected all food in ${s.steps} steps`;
  if (s.dead) return `Ran out of energy at step ${s.steps}`;
  return `Time ran out with ${s.food} food collected`;
}

export const CONTROLLER_LABEL: Record<string, string> = {
  untrained_population: "Untrained (10 random networks)",
  baseline: "Untrained baseline creature",
  random_actions: "Random actions",
  trained: "Trained creature",
  greedy_heuristic: "Hand-written greedy heuristic",
};

const ORDER = [
  "untrained_population",
  "baseline",
  "random_actions",
  "trained",
  "greedy_heuristic",
];

export function suiteRows(suite: Suite): string[][] {
  return ORDER.filter((k) => suite.controllers[k]).map((k) => {
    const m = suite.controllers[k];
    return [
      CONTROLLER_LABEL[k],
      `${m.food_mean.toFixed(2)} ± ${m.food_std.toFixed(2)}`,
      `${Math.round(m.success_rate * 100)}%`,
      m.survival_mean.toFixed(0),
      m.collisions_mean.toFixed(2),
      m.hazard_steps_mean.toFixed(1),
      m.reward_mean.toFixed(1),
    ];
  });
}

export const SUITE_HEADERS = [
  "Controller",
  "Food (of 6)",
  "Success",
  "Survival (steps)",
  "Collisions",
  "Hazard steps",
  "Reward",
];

export const VERDICT_TEXT: Record<string, string> = {
  improved:
    "Measured improvement: on held-out scenarios the trained creature beats the untrained population, with the 95% interval above zero for both food and reward.",
  no_clear_evidence:
    "No clear evidence of improvement: on held-out scenarios the 95% interval for trained minus untrained includes zero.",
  regressed:
    "Regression: on held-out scenarios the trained creature performs worse than the untrained population.",
};

export const VARIANT_LABEL: Record<string, string> = {
  standard: "Standard (training distribution)",
  dense_hazards: "Unseen: dense hazards",
  more_obstacles: "Unseen: more, larger obstacles",
  edge_food: "Unseen: food along the walls",
};
