"""Neuroevolution: a genetic algorithm over the weights of a fixed-topology network.

This is NOT NEAT: the topology never changes and there is no speciation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .config import TrainConfig
from .policy import GENOME_SIZE, NeuralPolicy
from .scenario import generate
from .world import World, run_episodes

BASELINE_INDEX = 0  # the untrained baseline is candidate 0 of the initial population


def initial_population(cfg: TrainConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(cfg.seed)
    return rng.normal(0.0, cfg.init_sigma, (cfg.population, GENOME_SIZE))


def next_generation(
    pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator, cfg: TrainConfig
) -> np.ndarray:
    """Elitism + tournament selection + uniform crossover + Gaussian mutation."""
    order = np.argsort(-fitness, kind="stable")
    children = [pop[i].copy() for i in order[: cfg.elite]]

    def pick() -> np.ndarray:
        rivals = rng.integers(len(pop), size=cfg.tournament)
        return pop[rivals[np.argmax(fitness[rivals])]]

    while len(children) < len(pop):
        child = pick().copy()
        if rng.random() < cfg.crossover_rate:
            other = pick()
            mask = rng.random(GENOME_SIZE) < 0.5
            child[mask] = other[mask]
        child += rng.normal(0.0, cfg.mutation_sigma, GENOME_SIZE)
        children.append(child)
    return np.array(children)


class PopulationEvaluator:
    """Runs every candidate on every training scenario through the shared engine."""

    def __init__(self, cfg: TrainConfig):
        self.n_scen = len(cfg.train_seeds)
        scenarios = [generate(s) for s in cfg.train_seeds]
        self.world = World(scenarios * cfg.population)

    def __call__(self, pop: np.ndarray) -> dict[str, np.ndarray]:
        w = self.world
        run_episodes(w, NeuralPolicy(np.repeat(pop, self.n_scen, axis=0)))
        per = lambda a: np.asarray(a, dtype=float).reshape(len(pop), self.n_scen).mean(axis=1)
        return {
            "fitness": per(w.rewards()),
            "food": per(w.food_count),
            "collisions": per(w.collisions),
            "survival": per(w.survival_steps()),
            "success": per(w.success),
        }


def train(
    cfg: TrainConfig,
    on_generation: Callable[[dict, np.ndarray], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Run the evolutionary loop. Generation 0 is the untouched random population."""
    rng = np.random.default_rng(cfg.seed)
    pop = initial_population(cfg, rng)
    baseline = pop[BASELINE_INDEX].copy()
    evaluate = PopulationEvaluator(cfg)
    history: list[dict] = []
    generation_best: list[np.ndarray] = []

    for gen in range(cfg.generations + 1):
        if gen > 0:
            pop = next_generation(pop, stats["fitness"], rng, cfg)
        stats = evaluate(pop)
        best = int(np.argmax(stats["fitness"]))
        record = {
            "generation": gen,
            "best_fitness": float(stats["fitness"][best]),
            "mean_fitness": float(stats["fitness"].mean()),
            "best_food": float(stats["food"][best]),
            "mean_food": float(stats["food"].mean()),
            "best_collisions": float(stats["collisions"][best]),
            "mean_collisions": float(stats["collisions"].mean()),
            "best_survival": float(stats["survival"][best]),
            "mean_survival": float(stats["survival"].mean()),
            "best_success": float(stats["success"][best]),
        }
        history.append(record)
        generation_best.append(pop[best].copy())
        if on_generation:
            on_generation(record, pop[best])
        if should_stop and should_stop():
            break

    return {
        "baseline": baseline,
        "best": generation_best[-1],
        "generation_best": generation_best,
        "history": history,
    }
