"""Held-out evaluation, unseen-scenario generalization and deterministic replay."""
from __future__ import annotations

import numpy as np

from .config import HELD_OUT_SEEDS, REWARD, UNSEEN_SEEDS, WORLD, TrainConfig
from .evolution import BASELINE_INDEX, initial_population
from .policy import GreedyFoodPolicy, NeuralPolicy, RandomActionPolicy
from .scenario import UNSEEN_VARIANTS, Scenario, generate
from .world import World, run_episodes

N_UNTRAINED = 10  # random-initial controllers averaged for the untrained reference
BOOTSTRAP_SAMPLES = 10_000
RANDOM_ACTION_SEED = 4242


def _episodes(scenarios: list[Scenario], policy) -> dict[str, np.ndarray]:
    w = World(scenarios)
    run_episodes(w, policy)
    return {
        "food": w.food_count.astype(float),
        "success": w.success.astype(float),
        "survival": w.survival_steps().astype(float),
        "steps": w.steps.astype(float),
        "collisions": w.collisions.astype(float),
        "hazard_steps": w.hazard_steps.astype(float),
        "reward": w.rewards(),
    }


def _summary(ep: dict[str, np.ndarray]) -> dict:
    return {
        "episodes": int(len(ep["food"])),
        "food_mean": float(ep["food"].mean()),
        "food_std": float(ep["food"].std()),
        "success_rate": float(ep["success"].mean()),
        "survival_mean": float(ep["survival"].mean()),
        "collisions_mean": float(ep["collisions"].mean()),
        "collisions_per_100_steps": float(100 * ep["collisions"].sum() / ep["steps"].sum()),
        "hazard_steps_mean": float(ep["hazard_steps"].mean()),
        "reward_mean": float(ep["reward"].mean()),
        "reward_std": float(ep["reward"].std()),
        "per_seed": {k: ep[k].round(4).tolist() for k in ("food", "reward")},
    }


def paired_difference(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Mean of a - b over matched seeds with a bootstrap 95% confidence interval."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
    }


def _suite(seeds, variant: str, baseline, trained, untrained) -> dict:
    scenarios = [generate(s, variant) for s in seeds]
    n = len(scenarios)
    eps = {
        "baseline": _episodes(scenarios, NeuralPolicy(np.tile(baseline, (n, 1)))),
        "trained": _episodes(scenarios, NeuralPolicy(np.tile(trained, (n, 1)))),
        "random_actions": _episodes(scenarios, RandomActionPolicy(RANDOM_ACTION_SEED)),
        "greedy_heuristic": _episodes(scenarios, GreedyFoodPolicy()),
    }
    # Every untrained controller faces every seed; then average per seed.
    pool = _episodes(scenarios * len(untrained), NeuralPolicy(np.repeat(untrained, n, axis=0)))
    controllers = {k: _summary(v) for k, v in eps.items()}
    controllers["untrained_population"] = _summary(pool)
    pool_seed = {k: v.reshape(len(untrained), n).mean(axis=0) for k, v in pool.items()}
    paired = {}
    for name, ref in (
        ("baseline", eps["baseline"]),
        ("untrained_population", pool_seed),
        ("greedy_heuristic", eps["greedy_heuristic"]),
    ):
        paired[f"trained_vs_{name}"] = {
            m: paired_difference(eps["trained"][m], ref[m]) for m in ("food", "reward")
        }
    return {"variant": variant, "seeds": list(seeds), "controllers": controllers, "paired": paired}


def learning_verdict(standard_suite: dict) -> dict:
    """Classify the held-out result. 'improved' requires both paired CIs above zero."""
    p = standard_suite["paired"]["trained_vs_untrained_population"]
    lows = [p[m]["ci95"][0] for m in ("food", "reward")]
    highs = [p[m]["ci95"][1] for m in ("food", "reward")]
    if min(lows) > 0:
        outcome = "improved"
    elif max(highs) < 0:
        outcome = "regressed"
    else:
        outcome = "no_clear_evidence"
    return {
        "outcome": outcome,
        "rule": "improved only if the bootstrap 95% CI of trained minus untrained-population "
        "is above zero for both food and reward on held-out standard scenarios",
    }


def evaluate(cfg: TrainConfig, baseline: np.ndarray, trained: np.ndarray) -> dict:
    """Full protocol: held-out in-distribution seeds plus unseen scenario variants."""
    untrained = initial_population(cfg)[:N_UNTRAINED]
    standard = _suite(HELD_OUT_SEEDS, "standard", baseline, trained, untrained)
    return {
        "protocol": {
            "train_seeds": list(cfg.train_seeds),
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "unseen_seeds": list(UNSEEN_SEEDS),
            "untrained_population_size": len(untrained),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "max_steps": WORLD.max_steps,
            "reward_weights": REWARD.__dict__,
        },
        "in_distribution": standard,
        "unseen": {
            v: _suite(UNSEEN_SEEDS, v, baseline, trained, untrained) for v in UNSEEN_VARIANTS
        },
        "verdict": learning_verdict(standard),
    }


def default_baseline() -> np.ndarray:
    """The untrained creature shown before any training run exists."""
    return initial_population(TrainConfig())[BASELINE_INDEX]


def replay(scenario: Scenario, controllers: list[tuple[str, object]]) -> dict:
    """Simulate named controllers on one identical scenario and record every step.

    `controllers` holds (name, genome) pairs; a genome of None means the greedy
    heuristic. Frame i of every replay is simulation step i, so clocks align.
    """
    replays = []
    for name, genome in controllers:
        single = World([scenario])
        policy = GreedyFoodPolicy() if genome is None else NeuralPolicy(genome)
        frames = run_episodes(single, policy, record=True)
        terms = {k: round(float(v[0]), 4) for k, v in single.reward_terms().items()}
        replays.append(
            {
                "controller": name,
                "x": [round(float(f[0][0, 0]), 4) for f in frames],
                "y": [round(float(f[0][0, 1]), 4) for f in frames],
                "energy": [round(float(max(f[1][0], 0.0)), 4) for f in frames],
                "food_left": [[bool(b) for b in f[2][0]] for f in frames],
                "contact": [bool(f[3][0]) for f in frames],
                "in_hazard": [bool(f[4][0]) for f in frames],
                "summary": {
                    "food": int(single.food_count[0]),
                    "success": bool(single.success[0]),
                    "dead": bool(single.dead[0]),
                    "steps": int(single.steps[0]),
                    "collisions": int(single.collisions[0]),
                    "hazard_steps": int(single.hazard_steps[0]),
                    "reward": round(float(single.rewards()[0]), 4),
                    "reward_terms": terms,
                },
            }
        )
    return {
        "world": {
            "max_steps": WORLD.max_steps,
            "creature_radius": WORLD.creature_radius,
            "food_radius": WORLD.food_radius,
            "sense_range": WORLD.sense_range,
        },
        "scenario": scenario.to_dict(),
        "replays": replays,
    }
