"""Controllers: the learnable neural policy and two non-learning reference policies."""
from __future__ import annotations

import numpy as np

from .world import ACT_SIZE, OBS_SIZE

HIDDEN = 12
_W1 = OBS_SIZE * HIDDEN
_B1 = _W1 + HIDDEN
_W2 = _B1 + HIDDEN * ACT_SIZE
GENOME_SIZE = _W2 + ACT_SIZE


class NeuralPolicy:
    """Fixed-topology MLP (14 -> 12 tanh -> 2 tanh), one genome per batch row."""

    def __init__(self, genomes: np.ndarray):
        g = np.atleast_2d(np.asarray(genomes, dtype=float))
        if g.shape[1] != GENOME_SIZE:
            raise ValueError(f"genome must have {GENOME_SIZE} parameters")
        n = len(g)
        self.w1 = g[:, :_W1].reshape(n, HIDDEN, OBS_SIZE)
        self.b1 = g[:, _W1:_B1]
        self.w2 = g[:, _B1:_W2].reshape(n, ACT_SIZE, HIDDEN)
        self.b2 = g[:, _W2:]

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Explicit sums (not BLAS) keep results identical for any batch size.
        hidden = np.tanh((self.w1 * obs[:, None, :]).sum(axis=2) + self.b1)
        return np.tanh((self.w2 * hidden[:, None, :]).sum(axis=2) + self.b2)


class GreedyFoodPolicy:
    """Hand-written heuristic: always accelerate straight at the nearest food."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, 2:4]


class RandomActionPolicy:
    """Uniform random accelerations, independent of observations."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, (len(obs), ACT_SIZE))
