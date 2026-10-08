export interface GenerationRecord {
  generation: number;
  best_fitness: number;
  mean_fitness: number;
  best_food: number;
  mean_food: number;
  best_collisions: number;
  mean_collisions: number;
  best_survival: number;
  mean_survival: number;
  best_success: number;
}

export interface ControllerMetrics {
  episodes: number;
  food_mean: number;
  food_std: number;
  success_rate: number;
  survival_mean: number;
  collisions_mean: number;
  collisions_per_100_steps: number;
  hazard_steps_mean: number;
  reward_mean: number;
  reward_std: number;
}

export interface Paired {
  mean: number;
  ci95: [number, number];
  wins: number;
  ties: number;
  losses: number;
}

export interface Suite {
  variant: string;
  seeds: number[];
  controllers: Record<string, ControllerMetrics>;
  paired: Record<string, { food: Paired; reward: Paired }>;
}

export interface Evaluation {
  in_distribution: Suite;
  unseen: Record<string, Suite>;
  verdict: { outcome: "improved" | "no_clear_evidence" | "regressed"; rule: string };
}

export type RunState = "running" | "completed" | "failed" | "interrupted";

export interface RunSummary {
  id: string;
  state: RunState;
  created: string;
  config: { population: number; generations: number; seed: number };
}

export interface Run extends RunSummary {
  generations: GenerationRecord[];
  evaluation: Evaluation | null;
}

export interface ReplaySummary {
  food: number;
  success: boolean;
  dead: boolean;
  steps: number;
  collisions: number;
  hazard_steps: number;
  reward: number;
}

export interface Replay {
  controller: string;
  x: number[];
  y: number[];
  energy: number[];
  food_left: boolean[][];
  contact: boolean[];
  in_hazard: boolean[];
  summary: ReplaySummary;
}

export interface Scenario {
  seed: number;
  variant: string;
  start: [number, number];
  food: [number, number][];
  obstacles: [number, number, number][];
  hazards: [number, number, number][];
}

export interface ReplayResponse {
  world: { max_steps: number; creature_radius: number; food_radius: number };
  scenario: Scenario;
  replays: Replay[];
}

export interface AppConfig {
  defaults: { population: number; generations: number; seed: number };
  variants: string[];
  held_out_seeds: number[];
  unseen_seeds: number[];
  max_steps: number;
}

export type ControllerRef =
  | { kind: "baseline" | "best" | "greedy" }
  | { kind: "generation"; generation: number };

export interface Api {
  config(): Promise<AppConfig>;
  runs(): Promise<RunSummary[]>;
  run(id: string): Promise<Run>;
  startTraining(cfg: Record<string, number>): Promise<string>;
  replay(
    runId: string | null,
    seed: number,
    variant: string,
    controllers: ControllerRef[],
  ): Promise<ReplayResponse>;
}
