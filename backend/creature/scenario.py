"""Deterministic scenario generation: a seed and a variant fully define a layout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WORLD

N_FOOD = 6

# "standard" is the training distribution. The other variants are never trained
# on and are used only for the unseen-scenario generalization check.
VARIANTS: dict[str, dict] = {
    "standard": dict(n_obstacles=4, obstacle_r=(0.045, 0.075), n_hazards=3, edge_food=False),
    "dense_hazards": dict(n_obstacles=4, obstacle_r=(0.045, 0.075), n_hazards=6, edge_food=False),
    "more_obstacles": dict(n_obstacles=7, obstacle_r=(0.055, 0.085), n_hazards=3, edge_food=False),
    "edge_food": dict(n_obstacles=4, obstacle_r=(0.045, 0.075), n_hazards=3, edge_food=True),
}
UNSEEN_VARIANTS = tuple(v for v in VARIANTS if v != "standard")
HAZARD_R = (0.06, 0.09)
MAX_OBSTACLES = max(v["n_obstacles"] for v in VARIANTS.values())
MAX_HAZARDS = max(v["n_hazards"] for v in VARIANTS.values())


@dataclass(frozen=True)
class Scenario:
    seed: int
    variant: str
    start: np.ndarray  # (2,)
    food: np.ndarray  # (N_FOOD, 2)
    obstacles: np.ndarray  # (k, 3) -> x, y, radius
    hazards: np.ndarray  # (k, 3) -> x, y, radius

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "variant": self.variant,
            "start": self.start.round(4).tolist(),
            "food": self.food.round(4).tolist(),
            "obstacles": self.obstacles.round(4).tolist(),
            "hazards": self.hazards.round(4).tolist(),
        }


def _surface_gap(p: np.ndarray, circles: list[np.ndarray]) -> float:
    """Smallest distance from point p to the surface of any circle (inf if none)."""
    if not circles:
        return float("inf")
    c = np.asarray(circles)
    return float((np.linalg.norm(c[:, :2] - p, axis=1) - c[:, 2]).min())


def _place(rng: np.random.Generator, sample, ok):
    for _ in range(20_000):
        item = sample()
        if ok(item):
            return item
    raise RuntimeError("could not place scenario element")


def generate(seed: int, variant: str = "standard") -> Scenario:
    if variant not in VARIANTS:
        raise ValueError(f"unknown scenario variant: {variant}")
    spec = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    cr = WORLD.creature_radius

    def circle(r_range, wall_gap):
        r = rng.uniform(*r_range)
        lo = r + wall_gap
        return np.array([rng.uniform(lo, 1 - lo), rng.uniform(lo, 1 - lo), r])

    obstacles: list[np.ndarray] = []
    for _ in range(spec["n_obstacles"]):
        obstacles.append(
            _place(
                rng,
                lambda: circle(spec["obstacle_r"], 0.06),
                lambda c: _surface_gap(c[:2], obstacles) - c[2] >= 0.07,
            )
        )

    hazards: list[np.ndarray] = []
    for _ in range(spec["n_hazards"]):
        hazards.append(
            _place(
                rng,
                lambda: circle(HAZARD_R, 0.0),
                lambda c: _surface_gap(c[:2], obstacles) - c[2] >= 0.02
                and _surface_gap(c[:2], hazards) - c[2] >= 0.05,
            )
        )

    start = _place(
        rng,
        lambda: rng.uniform(0.08, 0.92, 2),
        lambda p: _surface_gap(p, obstacles) >= 0.06 and _surface_gap(p, hazards) >= 0.06,
    )

    def food_point():
        if not spec["edge_food"]:
            return rng.uniform(0.06, 0.94, 2)
        p = rng.uniform(0.04, 0.96, 2)
        axis = rng.integers(2)
        band = rng.uniform(0.04, 0.14)
        p[axis] = band if rng.random() < 0.5 else 1 - band
        return p

    food: list[np.ndarray] = []
    for _ in range(N_FOOD):
        food.append(
            _place(
                rng,
                food_point,
                lambda p: _surface_gap(p, obstacles) >= cr + 0.03
                and _surface_gap(p, hazards) >= 0.035
                and np.linalg.norm(p - start) >= 0.15
                and all(np.linalg.norm(p - f) >= 0.1 for f in food),
            )
        )

    return Scenario(
        seed=seed,
        variant=variant,
        start=start,
        food=np.array(food),
        obstacles=np.array(obstacles).reshape(-1, 3),
        hazards=np.array(hazards).reshape(-1, 3),
    )
