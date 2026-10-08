import numpy as np
import pytest

from creature.config import REWARD, WORLD
from creature.policy import GENOME_SIZE, GreedyFoodPolicy, NeuralPolicy
from creature.scenario import VARIANTS, generate
from creature.world import OBS_SIZE, World, run_episodes

from conftest import make_scenario

RIGHT = np.array([[1.0, 0.0]])
IDLE = np.array([[0.0, 0.0]])


class IdlePolicy:
    def act(self, obs):
        return np.zeros((len(obs), 2))


def test_scenario_generation_is_deterministic():
    a, b = generate(7), generate(7)
    for field in ("start", "food", "obstacles", "hazards"):
        assert np.array_equal(getattr(a, field), getattr(b, field))
    assert not np.array_equal(a.food, generate(8).food)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_scenarios_are_well_formed(variant):
    for seed in range(20):
        s = generate(seed, variant)
        assert len(s.obstacles) == VARIANTS[variant]["n_obstacles"]
        assert len(s.hazards) == VARIANTS[variant]["n_hazards"]
        for f in s.food:
            assert (np.linalg.norm(s.hazards[:, :2] - f, axis=1) > s.hazards[:, 2]).all()
            assert (np.linalg.norm(s.obstacles[:, :2] - f, axis=1) > s.obstacles[:, 2]).all()
        assert ((s.food > 0) & (s.food < 1)).all()


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        generate(1, "lava")


def test_reset_is_deterministic_and_observation_bounded():
    w = World([generate(3), generate(4)])
    first = w.reset()
    for _ in range(30):
        w.step(np.ones((2, 2)))
    assert not np.array_equal(w.observe(), first)
    assert np.array_equal(w.reset(), first)
    assert first.shape == (2, OBS_SIZE)
    assert np.abs(first).max() <= 1.0 + 1e-9


def test_state_transition_matches_physics():
    w = World([make_scenario()])
    w.step(RIGHT)
    v = WORLD.max_accel * WORLD.damping
    assert w.vel[0] == pytest.approx([v, 0.0])
    assert w.pos[0] == pytest.approx([0.5 + v, 0.5])
    expected = 1.0 - WORLD.energy_base_drain - WORLD.energy_thrust_drain * 0.5
    assert w.energy[0] == pytest.approx(expected)
    for _ in range(100):
        w.step(RIGHT)
        assert np.linalg.norm(w.vel[0]) <= WORLD.max_speed + 1e-12


def test_actions_are_clipped():
    a, b = World([make_scenario()]), World([make_scenario()])
    a.step(np.array([[50.0, -50.0]]))
    b.step(np.array([[1.0, -1.0]]))
    assert np.array_equal(a.pos, b.pos)


def test_wall_collision_stops_and_counts_once():
    w = World([make_scenario(start=(0.9, 0.5), food=((0.1, 0.1),))])
    for _ in range(40):
        w.step(RIGHT)
    assert w.pos[0, 0] == pytest.approx(1.0 - WORLD.creature_radius)
    assert w.collisions[0] == 1


def test_obstacle_blocks_movement():
    w = World([make_scenario(start=(0.3, 0.5), food=((0.9, 0.5),), obstacles=((0.5, 0.5, 0.06),))])
    for _ in range(60):
        w.step(RIGHT)
    gap = np.linalg.norm(w.pos[0] - [0.5, 0.5])
    assert gap >= 0.06 + WORLD.creature_radius - 1e-9
    assert w.pos[0, 0] < 0.5
    assert w.collisions[0] >= 1
    assert w.food_count[0] == 0


def test_food_collection_restores_energy():
    w = World([make_scenario(start=(0.5, 0.5), food=((0.6, 0.5), (0.1, 0.1)))])
    for _ in range(30):
        before = w.energy[0]
        w.step(RIGHT)
        if w.food_count[0]:
            break
    assert w.food_count[0] == 1
    assert w.energy[0] > before
    assert w.energy[0] <= WORLD.energy_start
    assert w.food_left[0].tolist() == [False, True]


def test_hazard_drains_energy():
    safe = World([make_scenario()])
    hot = World([make_scenario(hazards=((0.5, 0.5, 0.1),))])
    safe.step(IDLE)
    hot.step(IDLE)
    assert hot.hazard_steps[0] == 1
    assert safe.energy[0] - hot.energy[0] == pytest.approx(WORLD.energy_hazard_drain)


def test_episode_ends_by_starvation():
    w = World([make_scenario()])
    run_episodes(w, IdlePolicy())
    assert w.dead[0] and not w.success[0]
    assert abs(w.steps[0] - 1.0 / WORLD.energy_base_drain) <= 1
    assert w.steps[0] < WORLD.max_steps
    frozen = w.pos.copy()
    w.step(RIGHT)
    assert np.array_equal(w.pos, frozen)


def test_episode_ends_by_success():
    w = World([make_scenario(food=((0.7, 0.5),))])
    run_episodes(w, GreedyFoodPolicy())
    assert w.success[0] and not w.dead[0]
    assert w.steps[0] < WORLD.max_steps
    assert w.survival_steps()[0] == WORLD.max_steps


def test_episode_never_exceeds_time_limit():
    w = World([generate(s) for s in range(6)])
    run_episodes(w, GreedyFoodPolicy())
    assert w.done.all() and w.t <= WORLD.max_steps
    assert (w.steps <= WORLD.max_steps).all()


def test_reward_is_sum_of_measured_terms():
    w = World([generate(11)])
    run_episodes(w, GreedyFoodPolicy())
    terms = w.reward_terms()
    assert terms["food"][0] == REWARD.food * w.food_count[0]
    assert terms["collision"][0] == REWARD.collision * w.collisions[0]
    assert terms["hazard"][0] == REWARD.hazard_step * w.hazard_steps[0]
    assert w.rewards()[0] == pytest.approx(sum(t[0] for t in terms.values()))


def test_progress_cannot_be_farmed_by_oscillating():
    w = World([make_scenario(start=(0.5, 0.5), food=((0.9, 0.5),))])
    for i in range(200):
        w.step(RIGHT if (i // 10) % 2 == 0 else -RIGHT)
    assert w.food_count[0] == 0
    assert w.progress[0] <= 0.4  # can never exceed the initial distance to the food


def test_batch_rows_match_single_simulation():
    scenarios = [generate(s) for s in (1, 2, 3)]
    genomes = np.random.default_rng(0).normal(0, 0.5, (3, GENOME_SIZE))
    batch = World(scenarios)
    run_episodes(batch, NeuralPolicy(genomes))
    for i, s in enumerate(scenarios):
        single = World([s])
        run_episodes(single, NeuralPolicy(genomes[i]))
        assert np.array_equal(single.pos[0], batch.pos[i])
        assert single.rewards()[0] == batch.rewards()[i]
