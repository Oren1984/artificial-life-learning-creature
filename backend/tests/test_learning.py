import numpy as np
import pytest
from pydantic import ValidationError

from creature.config import HELD_OUT_SEEDS, RESERVED_EVAL_SEEDS, UNSEEN_SEEDS, TrainConfig
from creature.evaluation import evaluate, learning_verdict, paired_difference, replay
from creature.evolution import PopulationEvaluator, initial_population, next_generation, train
from creature.policy import GENOME_SIZE, GreedyFoodPolicy, NeuralPolicy
from creature.scenario import generate
from creature.world import OBS_SIZE, World, run_episodes


def test_controller_inference_shape_bounds_and_determinism():
    rng = np.random.default_rng(1)
    genomes = rng.normal(0, 1, (5, GENOME_SIZE))
    obs = rng.uniform(-1, 1, (5, OBS_SIZE))
    out = NeuralPolicy(genomes).act(obs)
    assert out.shape == (5, 2)
    assert np.abs(out).max() <= 1.0
    assert np.array_equal(out, NeuralPolicy(genomes).act(obs))
    assert np.array_equal(out[2], NeuralPolicy(genomes[2]).act(obs[2:3])[0])
    assert np.array_equal(NeuralPolicy(np.zeros(GENOME_SIZE)).act(obs[:1]), np.zeros((1, 2)))


def test_controller_rejects_wrong_genome_size():
    with pytest.raises(ValueError):
        NeuralPolicy(np.zeros(GENOME_SIZE + 1))


def test_mutation_changes_children_but_preserves_elites(small_cfg):
    rng = np.random.default_rng(0)
    pop = initial_population(small_cfg)
    fitness = np.arange(len(pop), dtype=float)
    new = next_generation(pop, fitness, rng, small_cfg)
    assert new.shape == pop.shape
    assert np.array_equal(new[0], pop[-1])  # best
    assert np.array_equal(new[1], pop[-2])  # second best
    for child in new[small_cfg.elite :]:
        assert not any(np.array_equal(child, parent) for parent in pop)


def test_selection_favours_fitter_parents(small_cfg):
    pop = np.zeros((100, GENOME_SIZE))
    pop[50:] = 100.0  # the fit half carries a recognizable genome
    fitness = np.repeat([0.0, 1.0], 50)
    cfg = small_cfg.model_copy(update={"elite": 0, "crossover_rate": 0.0})
    new = next_generation(pop, fitness, np.random.default_rng(0), cfg)
    # Tournament of 3 picks a fit parent with p = 1 - 0.5**3 = 0.875, not 0.5.
    assert (new.mean(axis=1) > 50).mean() > 0.7


def test_training_is_reproducible_and_updates_parameters(small_cfg):
    a, b = train(small_cfg), train(small_cfg)
    assert a["history"] == b["history"]
    assert np.array_equal(a["best"], b["best"])
    assert not np.array_equal(a["best"], a["baseline"])
    assert np.array_equal(a["baseline"], initial_population(small_cfg)[0])
    other = train(small_cfg.model_copy(update={"seed": 1}))
    assert not np.array_equal(other["best"], a["best"])


def test_reported_fitness_comes_from_simulation(small_cfg):
    result = train(small_cfg)
    assert len(result["history"]) == small_cfg.generations + 1
    scenarios = [generate(s) for s in small_cfg.train_seeds]
    for gen in (0, small_cfg.generations):
        genome = result["generation_best"][gen]
        w = World(scenarios)
        run_episodes(w, NeuralPolicy(np.tile(genome, (len(scenarios), 1))))
        record = result["history"][gen]
        assert record["best_fitness"] == pytest.approx(w.rewards().mean(), abs=1e-12)
        assert record["best_food"] == pytest.approx(w.food_count.mean())
        assert record["best_collisions"] == pytest.approx(w.collisions.mean())


def test_generation_zero_is_the_unmodified_random_population(small_cfg):
    stats = PopulationEvaluator(small_cfg)(initial_population(small_cfg))
    record = train(small_cfg)["history"][0]
    assert record["best_fitness"] == pytest.approx(stats["fitness"].max())
    assert record["mean_fitness"] == pytest.approx(stats["fitness"].mean())


def test_elitism_never_loses_training_fitness(small_cfg):
    best = [r["best_fitness"] for r in train(small_cfg)["history"]]
    assert all(later >= earlier for earlier, later in zip(best, best[1:]))


def test_evaluation_seeds_are_separated_from_training():
    assert not set(TrainConfig().train_seeds) & RESERVED_EVAL_SEEDS
    assert not set(HELD_OUT_SEEDS) & set(UNSEEN_SEEDS)
    for leaked in (HELD_OUT_SEEDS[0], UNSEEN_SEEDS[-1]):
        with pytest.raises(ValidationError):
            TrainConfig(train_seeds=[100, leaked])


@pytest.mark.parametrize(
    "bad",
    [
        dict(population=2),
        dict(generations=0),
        dict(population=200, generations=300),
        dict(train_seeds=[]),
        dict(train_seeds=[1, 1]),
        dict(elite=40),
        dict(mutation_sigma=0),
        dict(unknown_field=1),
    ],
)
def test_train_config_validation(bad):
    with pytest.raises(ValidationError):
        TrainConfig(**bad)


def test_evaluation_uses_the_specified_controllers(small_cfg):
    weak = initial_population(small_cfg)[0]
    result = evaluate(small_cfg, weak, weak)
    suite = result["in_distribution"]
    assert suite["seeds"] == list(HELD_OUT_SEEDS)
    base, trained = suite["controllers"]["baseline"], suite["controllers"]["trained"]
    assert base == trained  # same genome in, same measurements out
    assert suite["paired"]["trained_vs_baseline"]["food"]["mean"] == 0
    scenarios = [generate(s) for s in HELD_OUT_SEEDS]
    w = World(scenarios)
    run_episodes(w, NeuralPolicy(np.tile(weak, (len(scenarios), 1))))
    assert trained["food_mean"] == pytest.approx(w.food_count.mean())
    assert trained["reward_mean"] == pytest.approx(w.rewards().mean())
    g = World(scenarios)
    run_episodes(g, GreedyFoodPolicy())
    assert suite["controllers"]["greedy_heuristic"]["food_mean"] == pytest.approx(g.food_count.mean())
    assert set(result["unseen"]) == {"dense_hazards", "more_obstacles", "edge_food"}
    assert result["unseen"]["edge_food"]["seeds"] == list(UNSEEN_SEEDS)


def test_failed_learning_is_not_reported_as_success(small_cfg):
    untrained = initial_population(small_cfg)[0]
    result = evaluate(small_cfg, untrained, untrained)
    assert result["verdict"]["outcome"] != "improved"


def _suite(food_ci, reward_ci):
    pair = {"food": {"ci95": food_ci}, "reward": {"ci95": reward_ci}}
    return {"paired": {"trained_vs_untrained_population": pair}}


def test_verdict_rule():
    assert learning_verdict(_suite([0.5, 2], [1, 9]))["outcome"] == "improved"
    assert learning_verdict(_suite([-0.1, 2], [1, 9]))["outcome"] == "no_clear_evidence"
    assert learning_verdict(_suite([0.5, 2], [-3, 9]))["outcome"] == "no_clear_evidence"
    assert learning_verdict(_suite([-2, -1], [-9, -1]))["outcome"] == "regressed"


def test_paired_difference():
    d = paired_difference(np.array([3.0, 4, 5, 6]), np.array([1.0, 4, 2, 7]))
    assert d["mean"] == pytest.approx(1.0)
    assert (d["wins"], d["ties"], d["losses"]) == (2, 1, 1)
    assert d["ci95"][0] <= d["mean"] <= d["ci95"][1]
    assert paired_difference(np.ones(5), np.ones(5))["ci95"] == [0.0, 0.0]


def test_replay_matches_simulation_and_is_deterministic(small_cfg):
    result = train(small_cfg)
    scenario = generate(HELD_OUT_SEEDS[0])
    pair = [("baseline", result["baseline"]), ("best", result["best"])]
    out = replay(scenario, pair)
    assert out == replay(scenario, pair)
    assert out["scenario"]["seed"] == HELD_OUT_SEEDS[0]
    for rep, genome in zip(out["replays"], (result["baseline"], result["best"])):
        w = World([scenario])
        run_episodes(w, NeuralPolicy(genome))
        assert rep["summary"]["food"] == w.food_count[0]
        assert rep["summary"]["steps"] == w.steps[0] == len(rep["x"]) - 1
        assert rep["summary"]["reward"] == pytest.approx(w.rewards()[0], abs=1e-4)
        assert rep["x"][0] == pytest.approx(scenario.start[0], abs=1e-4)
        assert sum(rep["food_left"][-1]) == 6 - w.food_count[0]
