import { vi } from "vitest";
import type {
  Api,
  AppConfig,
  ControllerMetrics,
  ControllerRef,
  Evaluation,
  GenerationRecord,
  Replay,
  ReplayResponse,
  Run,
  Suite,
} from "../src/types";

export const CONFIG: AppConfig = {
  defaults: { population: 40, generations: 50, seed: 0 },
  variants: ["standard", "dense_hazards"],
  held_out_seeds: [9000, 9001],
  unseen_seeds: [9100, 9101],
  max_steps: 500,
};

export function gen(generation: number, best: number, mean: number): GenerationRecord {
  return {
    generation,
    best_fitness: best,
    mean_fitness: mean,
    best_food: 2,
    mean_food: 1,
    best_collisions: 1,
    mean_collisions: 2,
    best_survival: 400,
    mean_survival: 300,
    best_success: 0,
  };
}

export function makeReplay(controller: string, food: number, steps = 4): Replay {
  const frames = steps + 1;
  return {
    controller,
    x: Array.from({ length: frames }, (_, i) => 0.1 + 0.1 * i),
    y: Array.from({ length: frames }, () => 0.5),
    energy: Array.from({ length: frames }, (_, i) => 1 - 0.1 * i),
    food_left: Array.from({ length: frames }, (_, i) => [i < 2 || food === 0, true]),
    contact: Array.from({ length: frames }, (_, i) => i === 3),
    in_hazard: Array.from({ length: frames }, (_, i) => i >= 3),
    summary: {
      food,
      success: false,
      dead: true,
      steps,
      collisions: 1,
      hazard_steps: 2,
      reward: food * 10,
    },
  };
}

export function replayResponse(controllers: string[]): ReplayResponse {
  return {
    world: { max_steps: 500, creature_radius: 0.018, food_radius: 0.014 },
    scenario: {
      seed: 9000,
      variant: "standard",
      start: [0.1, 0.5],
      food: [
        [0.3, 0.5],
        [0.9, 0.9],
      ],
      obstacles: [[0.5, 0.2, 0.05]],
      hazards: [[0.7, 0.7, 0.08]],
    },
    replays: controllers.map((c, i) => makeReplay(c, i)),
  };
}

function metrics(food: number): ControllerMetrics {
  return {
    episodes: 30,
    food_mean: food,
    food_std: 0.5,
    success_rate: food / 6,
    survival_mean: 300,
    collisions_mean: 2,
    collisions_per_100_steps: 0.5,
    hazard_steps_mean: 3,
    reward_mean: food * 10,
    reward_std: 4,
  };
}

function suite(variant: string): Suite {
  const pair = { mean: 5.3, ci95: [4.8, 5.7] as [number, number], wins: 30, ties: 0, losses: 0 };
  return {
    variant,
    seeds: [9000, 9001],
    controllers: {
      baseline: metrics(0.2),
      trained: metrics(5.5),
      untrained_population: metrics(0.3),
      random_actions: metrics(0.1),
      greedy_heuristic: metrics(5.8),
    },
    paired: { trained_vs_untrained_population: { food: pair, reward: pair } },
  };
}

export function evaluation(outcome: Evaluation["verdict"]["outcome"]): Evaluation {
  return {
    in_distribution: suite("standard"),
    unseen: { dense_hazards: suite("dense_hazards") },
    verdict: { outcome, rule: "rule" },
  };
}

export function makeRun(state: Run["state"], generations: GenerationRecord[]): Run {
  return {
    id: "20261008-120000-abcdef",
    state,
    created: "2026-10-08T12:00:00+00:00",
    config: { population: 40, generations: 50, seed: 0 },
    generations,
    evaluation: state === "completed" ? evaluation("improved") : null,
  };
}

/** Fake backend whose current run can be swapped while the app polls. */
export function fakeApi(initial: Run | null) {
  const state = { run: initial };
  const api: Api = {
    config: vi.fn(async () => CONFIG),
    runs: vi.fn(async () => (state.run ? [state.run] : [])),
    run: vi.fn(async () => {
      if (!state.run) throw new Error("run not found");
      return state.run;
    }),
    startTraining: vi.fn(async () => {
      state.run = makeRun("running", [gen(0, 5, 1)]);
      return state.run.id;
    }),
    replay: vi.fn(async (_run, _seed, _variant, controllers) =>
      replayResponse(controllers.map((c: ControllerRef) => c.kind)),
    ),
  };
  return { api, state };
}
