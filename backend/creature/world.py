"""The single authoritative simulation engine.

`World` simulates a batch of independent creatures, one per scenario row.
Training, evaluation and replay all go through this class, so there is exactly
one implementation of the physics, sensing and reward rules.

Observation vector (14 values, all normalized):
    0-1   velocity / max_speed
    2-3   unit direction to the nearest remaining food
    4     distance to that food (clipped to 1)
    5-6   unit direction to the nearest obstacle (zero if out of sense range)
    7     obstacle proximity in [0, 1]
    8-9   unit direction to the nearest hazard (zero if out of sense range)
    10    hazard proximity in [0, 1] (1 = inside)
    11-12 position mapped to [-1, 1]
    13    remaining energy in [0, 1]

Action vector (2 values in [-1, 1]): horizontal and vertical acceleration.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .config import REWARD, WORLD, RewardConfig, WorldConfig
from .scenario import MAX_HAZARDS, MAX_OBSTACLES, Scenario

OBS_SIZE = 14
ACT_SIZE = 2
_FAR = np.array([-10.0, -10.0, 0.0])  # padding circle: never sensed, never hit


class Policy(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray: ...


def _pad(circles: np.ndarray, n: int) -> np.ndarray:
    out = np.tile(_FAR, (n, 1))
    out[: len(circles)] = circles
    return out


class World:
    def __init__(self, scenarios: Sequence[Scenario], cfg: WorldConfig = WORLD):
        self.cfg = cfg
        self.n = len(scenarios)
        self._start = np.stack([s.start for s in scenarios])
        self.food = np.stack([s.food for s in scenarios])
        self.obstacles = np.stack([_pad(s.obstacles, MAX_OBSTACLES) for s in scenarios])
        self.hazards = np.stack([_pad(s.hazards, MAX_HAZARDS) for s in scenarios])
        self._rows = np.arange(self.n)
        self.reset()

    def reset(self) -> np.ndarray:
        n = self.n
        self.t = 0
        self.pos = self._start.copy()
        self.vel = np.zeros((n, 2))
        self.energy = np.full(n, self.cfg.energy_start)
        self.food_left = np.ones(self.food.shape[:2], dtype=bool)
        self.done = np.zeros(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        self.dead = np.zeros(n, dtype=bool)
        self.contact = np.zeros(n, dtype=bool)
        self.in_hazard = np.zeros(n, dtype=bool)
        self.steps = np.zeros(n, dtype=int)
        self.food_count = np.zeros(n, dtype=int)
        self.collisions = np.zeros(n, dtype=int)
        self.hazard_steps = np.zeros(n, dtype=int)
        self.progress = np.zeros(n)
        self.effort = np.zeros(n)
        return self.observe()

    # -- sensing ---------------------------------------------------------
    def _nearest_food(self):
        vec = self.food - self.pos[:, None, :]
        dist = np.where(self.food_left, np.linalg.norm(vec, axis=2), np.inf)
        idx = dist.argmin(axis=1)
        nearest = dist[self._rows, idx]
        has = np.isfinite(nearest)
        nearest = np.where(has, nearest, 0.0)
        unit = vec[self._rows, idx] / np.maximum(nearest, 1e-9)[:, None]
        return idx, nearest, np.where(has[:, None], unit, 0.0), has

    def _nearest_circle(self, circles: np.ndarray, body: float):
        vec = circles[:, :, :2] - self.pos[:, None, :]
        dist = np.linalg.norm(vec, axis=2)
        gap = dist - circles[:, :, 2] - body
        idx = gap.argmin(axis=1)
        prox = np.clip(1.0 - gap[self._rows, idx] / self.cfg.sense_range, 0.0, 1.0)
        unit = vec[self._rows, idx] / np.maximum(dist[self._rows, idx], 1e-9)[:, None]
        return np.where((prox > 0)[:, None], unit, 0.0), prox

    def observe(self) -> np.ndarray:
        c = self.cfg
        _, food_dist, food_dir, _ = self._nearest_food()
        obs_dir, obs_prox = self._nearest_circle(self.obstacles, c.creature_radius)
        haz_dir, haz_prox = self._nearest_circle(self.hazards, 0.0)
        return np.concatenate(
            [
                self.vel / c.max_speed,
                food_dir,
                np.minimum(food_dist, 1.0)[:, None],
                obs_dir,
                obs_prox[:, None],
                haz_dir,
                haz_prox[:, None],
                self.pos * 2.0 - 1.0,
                np.clip(self.energy, 0.0, 1.0)[:, None],
            ],
            axis=1,
        )

    # -- dynamics --------------------------------------------------------
    def step(self, actions: np.ndarray) -> None:
        c = self.cfg
        act = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
        active = ~self.done
        target, d_before, _, has_food = self._nearest_food()

        vel = (self.vel + act * c.max_accel) * c.damping
        speed = np.linalg.norm(vel, axis=1)
        vel = vel * np.minimum(1.0, c.max_speed / np.maximum(speed, 1e-12))[:, None]
        pos = self.pos + vel

        lo, hi = c.creature_radius, 1.0 - c.creature_radius
        contact = np.zeros(self.n, dtype=bool)
        for k in range(self.obstacles.shape[1]):
            center = self.obstacles[:, k, :2]
            reach = self.obstacles[:, k, 2] + c.creature_radius
            off = pos - center
            dist = np.linalg.norm(off, axis=1)
            over = dist < reach
            normal = off / np.maximum(dist, 1e-9)[:, None]
            pos = np.where(over[:, None], center + normal * reach[:, None], pos)
            vn = (vel * normal).sum(axis=1)
            vel = np.where((over & (vn < 0))[:, None], vel - vn[:, None] * normal, vel)
            contact |= over
        wall = (pos < lo) | (pos > hi)
        pos = np.clip(pos, lo, hi)
        vel = np.where(wall, 0.0, vel)
        contact |= wall.any(axis=1)

        new_hit = contact & ~self.contact & active
        haz_dist = np.linalg.norm(self.hazards[:, :, :2] - pos[:, None, :], axis=2)
        in_hazard = (haz_dist < self.hazards[:, :, 2]).any(axis=1) & active
        effort = (act**2).sum(axis=1) / 2.0

        food_dist = np.linalg.norm(self.food - pos[:, None, :], axis=2)
        eaten = self.food_left & active[:, None] & (food_dist < c.food_radius + c.creature_radius)
        n_eaten = eaten.sum(axis=1)

        energy = (
            self.energy
            - c.energy_base_drain
            - c.energy_thrust_drain * effort
            - c.energy_hazard_drain * in_hazard
            - c.energy_collision_cost * new_hit
        )
        energy = np.minimum(c.energy_start, energy + c.energy_food_gain * n_eaten)

        # Net approach to the food that was nearest before the step. Summed over
        # an episode this is bounded by the distances travelled between foods,
        # so it cannot be farmed by circling or oscillating.
        d_after = np.linalg.norm(self.food[self._rows, target] - pos, axis=1)
        gain = np.where(has_food, d_before - d_after, 0.0)

        m = active
        self.pos = np.where(m[:, None], pos, self.pos)
        self.vel = np.where(m[:, None], vel, self.vel)
        self.energy = np.where(m, energy, self.energy)
        self.contact = np.where(m, contact, self.contact)
        self.in_hazard = in_hazard
        self.food_left &= ~eaten
        self.steps += m
        self.food_count += n_eaten
        self.collisions += new_hit
        self.hazard_steps += in_hazard
        self.progress += np.where(m, gain, 0.0)
        self.effort += np.where(m, effort, 0.0)

        self.t += 1
        finished = ~self.food_left.any(axis=1)
        self.success |= m & finished
        self.dead |= m & ~finished & (self.energy <= 0.0)
        self.done |= self.success | self.dead | (self.t >= c.max_steps)

    # -- outcomes --------------------------------------------------------
    def survival_steps(self) -> np.ndarray:
        """Steps survived; finishing all food counts as surviving the full episode."""
        return np.where(self.dead, self.steps, self.cfg.max_steps)

    def reward_terms(self, weights: RewardConfig = REWARD) -> dict[str, np.ndarray]:
        """Fitness terms per episode, computed only from simulation outcomes."""
        T = self.cfg.max_steps
        return {
            "food": weights.food * self.food_count,
            "progress": weights.progress * self.progress,
            "completion": weights.completion * self.success * (1.0 + (T - self.steps) / T),
            "survival": weights.survival * self.survival_steps() / T,
            "collision": weights.collision * self.collisions,
            "hazard": weights.hazard_step * self.hazard_steps,
            "effort": weights.effort * self.effort / T,
        }

    def rewards(self) -> np.ndarray:
        return sum(self.reward_terms().values())


def run_episodes(world: World, policy: Policy, record: bool = False):
    """Run every row of `world` to termination. Returns recorded frames if asked."""
    obs = world.reset()
    frames = [] if record else None

    def snap():
        frames.append(
            (
                world.pos.copy(),
                world.energy.copy(),
                world.food_left.copy(),
                world.contact.copy(),
                world.in_hazard.copy(),
            )
        )

    if record:
        snap()
    while not world.done.all():
        world.step(policy.act(obs))
        obs = world.observe()
        if record:
            snap()
    return frames
