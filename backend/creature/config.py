"""Explicit configuration: world rules, reward weights, seeds and training settings.

World rules and reward weights are frozen constants so that every candidate,
generation and evaluation episode is simulated under identical rules. Only the
training budget (TrainConfig) is user-configurable, and it is validated.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class WorldConfig:
    """Physics and energy rules of the unit-square world (one step = one tick)."""

    max_steps: int = 500
    creature_radius: float = 0.018
    food_radius: float = 0.014
    max_accel: float = 0.003
    max_speed: float = 0.02
    damping: float = 0.93
    sense_range: float = 0.3
    energy_start: float = 1.0
    energy_base_drain: float = 0.003
    energy_thrust_drain: float = 0.002
    energy_hazard_drain: float = 0.08
    energy_collision_cost: float = 0.02
    energy_food_gain: float = 0.3


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the individually reported fitness terms."""

    food: float = 10.0
    progress: float = 5.0
    completion: float = 5.0
    survival: float = 2.0
    collision: float = -0.5
    hazard_step: float = -0.2
    effort: float = -1.0


WORLD = WorldConfig()
REWARD = RewardConfig()

# Scenario seeds. Held-out seeds are reserved for evaluation and can never be
# used for training (enforced by TrainConfig validation).
DEFAULT_TRAIN_SEEDS = tuple(range(100, 108))
HELD_OUT_SEEDS = tuple(range(9000, 9030))
UNSEEN_SEEDS = tuple(range(9100, 9130))
RESERVED_EVAL_SEEDS = frozenset(HELD_OUT_SEEDS) | frozenset(UNSEEN_SEEDS)

# Upper bound on episodes simulated by one training job (CPU guard).
MAX_TRAINING_EPISODES = 200_000


class TrainConfig(BaseModel):
    """Training budget and evolutionary hyperparameters."""

    population: int = Field(40, ge=4, le=200)
    generations: int = Field(50, ge=1, le=300)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    train_seeds: list[int] = Field(
        default_factory=lambda: list(DEFAULT_TRAIN_SEEDS), min_length=1, max_length=16
    )
    elite: int = Field(2, ge=0, le=20)
    tournament: int = Field(3, ge=2, le=10)
    crossover_rate: float = Field(0.5, ge=0.0, le=1.0)
    mutation_sigma: float = Field(0.1, gt=0.0, le=2.0)
    init_sigma: float = Field(0.5, gt=0.0, le=5.0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> "TrainConfig":
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("train_seeds must be unique")
        if any(s < 0 or s > 2**31 - 1 for s in self.train_seeds):
            raise ValueError("train_seeds must be non-negative 31-bit integers")
        leaked = sorted(set(self.train_seeds) & RESERVED_EVAL_SEEDS)
        if leaked:
            raise ValueError(f"train_seeds overlap reserved evaluation seeds: {leaked}")
        if self.elite >= self.population:
            raise ValueError("elite must be smaller than population")
        episodes = self.population * len(self.train_seeds) * (self.generations + 1)
        if episodes > MAX_TRAINING_EPISODES:
            raise ValueError(
                f"training budget too large: {episodes} episodes > {MAX_TRAINING_EPISODES}"
            )
        return self
