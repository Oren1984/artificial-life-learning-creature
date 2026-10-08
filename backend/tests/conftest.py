import numpy as np
import pytest

from creature.config import TrainConfig
from creature.scenario import Scenario

SMALL = dict(population=8, generations=3, train_seeds=[100, 101])


@pytest.fixture
def small_cfg():
    return TrainConfig(**SMALL)


def make_scenario(start=(0.5, 0.5), food=((0.9, 0.9),), obstacles=(), hazards=()):
    """Hand-built layout so each rule can be tested in isolation."""
    return Scenario(
        seed=-1,
        variant="test",
        start=np.array(start, dtype=float),
        food=np.array(food, dtype=float).reshape(-1, 2),
        obstacles=np.array(obstacles, dtype=float).reshape(-1, 3),
        hazards=np.array(hazards, dtype=float).reshape(-1, 3),
    )
