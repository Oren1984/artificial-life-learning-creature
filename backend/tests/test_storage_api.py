import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from creature.api import create_app
from creature.config import HELD_OUT_SEEDS, TrainConfig
from creature.evolution import train
from creature.jobs import run_training, start_run
from creature.policy import NeuralPolicy
from creature.storage import RunNotFound, RunStore

from conftest import SMALL

GHOST = "20260101-000000-abcdef"


@pytest.fixture
def client(tmp_path):
    app = create_app(artifacts_dir=tmp_path, frontend_dir=tmp_path / "missing")
    with TestClient(app) as c:
        c.manager = app.state.manager
        yield c


def _train(client, **overrides):
    r = client.post("/api/training", json={**SMALL, **overrides})
    assert r.status_code == 202, r.text
    client.manager.wait(120)
    return r.json()["run_id"]


def test_checkpoint_roundtrip_reproduces_policy(tmp_path, small_cfg):
    store = RunStore(tmp_path)
    run_id = start_run(store, small_cfg)
    run_training(store, run_id, small_cfg)
    direct = train(small_cfg)
    reopened = RunStore(tmp_path)  # fresh handle = "after a restart"
    loaded = reopened.genome(run_id, "best")
    assert np.array_equal(loaded, direct["best"])
    assert np.array_equal(reopened.genome(run_id, "baseline"), direct["baseline"])
    assert np.array_equal(reopened.genome(run_id, "generation", 1), direct["generation_best"][1])
    obs = np.random.default_rng(0).uniform(-1, 1, (4, 14))
    tile = lambda g: np.tile(g, (4, 1))
    assert np.array_equal(
        NeuralPolicy(tile(loaded)).act(obs), NeuralPolicy(tile(direct["best"])).act(obs)
    )
    assert reopened.read(run_id, "metrics")["generations"] == direct["history"]
    assert reopened.read(run_id, "run")["state"] == "completed"
    assert not list((tmp_path / "runs" / run_id).glob("*.tmp"))


def test_store_rejects_bad_ids_and_marks_interrupted_runs(tmp_path):
    store = RunStore(tmp_path)
    for bad in ("../../etc", "nope", "20260101-000000-zzzzzz"):
        with pytest.raises(RunNotFound):
            store.read(bad, "run")
    run_id = store.create({})
    store.mark_interrupted()
    assert store.read(run_id, "run")["state"] == "interrupted"


def test_health_and_config(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    cfg = client.get("/api/config").json()
    assert cfg["defaults"] == TrainConfig().model_dump()
    assert cfg["held_out_seeds"] == list(HELD_OUT_SEEDS)
    assert "standard" in cfg["variants"]


def test_training_run_lifecycle(client):
    assert client.get("/api/runs").json() == {"runs": []}
    run_id = _train(client)
    assert client.get("/api/training/status").json() == {"busy": False, "run_id": run_id}
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["state"] == "completed"
    assert [g["generation"] for g in run["generations"]] == [0, 1, 2, 3]
    assert run["generations"] == train(TrainConfig(**SMALL))["history"]
    assert run["evaluation"]["verdict"]["outcome"] in {"improved", "no_clear_evidence", "regressed"}
    assert client.get("/api/runs").json()["runs"][0]["id"] == run_id
    again = client.post(f"/api/runs/{run_id}/evaluate")
    assert again.status_code == 200
    assert again.json() == run["evaluation"]


def test_replay_endpoint(client):
    run_id = _train(client)
    body = {
        "run_id": run_id,
        "seed": 9001,
        "variant": "dense_hazards",
        "controllers": [
            {"kind": "baseline"},
            {"kind": "best"},
            {"kind": "generation", "generation": 1},
        ],
    }
    r = client.post("/api/replay", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["scenario"]["variant"] == "dense_hazards"
    assert len(data["scenario"]["hazards"]) == 6
    assert [x["controller"] for x in data["replays"]] == ["baseline", "best", "generation 1"]
    assert client.post("/api/replay", json=body).json() == data
    last = {**body, "controllers": [{"kind": "generation", "generation": 3}]}
    final = client.post("/api/replay", json=last).json()
    assert final["replays"][0]["x"] == data["replays"][1]["x"]  # last generation's best == best


def test_replay_without_a_run_only_offers_the_baseline(client):
    ok = client.post("/api/replay", json={"controllers": [{"kind": "baseline"}, {"kind": "greedy"}]})
    assert ok.status_code == 200
    assert len(ok.json()["replays"]) == 2
    assert client.post("/api/replay", json={"controllers": [{"kind": "best"}]}).status_code == 400


def test_api_validation_and_errors(client):
    assert client.post("/api/training", json={"population": 100000}).status_code == 422
    assert client.post("/api/training", json={"train_seeds": [9000]}).status_code == 422
    assert client.post("/api/training", json={"shell": "rm -rf"}).status_code == 422
    assert client.get(f"/api/runs/{GHOST}").status_code == 404
    assert client.get("/api/runs/not-a-run").status_code == 404
    assert client.post(f"/api/runs/{GHOST}/evaluate").status_code == 404
    for bad in (
        {"controllers": []},
        {"controllers": [{"kind": "script"}]},
        {"controllers": [{"kind": "baseline"}], "variant": "lava"},
        {"controllers": [{"kind": "baseline"}], "seed": -1},
        {"controllers": [{"kind": "baseline"}], "run_id": "../secrets"},
    ):
        assert client.post("/api/replay", json=bad).status_code == 422
    missing = {"run_id": GHOST, "controllers": [{"kind": "best"}]}
    assert client.post("/api/replay", json=missing).status_code == 404
    run_id = _train(client)
    far = {"run_id": run_id, "controllers": [{"kind": "generation", "generation": 99}]}
    assert client.post("/api/replay", json=far).status_code == 404


def test_errors_do_not_leak_paths(client, tmp_path):
    responses = (
        client.get(f"/api/runs/{GHOST}"),
        client.post("/api/replay", json={"run_id": "x", "controllers": []}),
    )
    for r in responses:
        assert str(tmp_path) not in r.text
        assert "Traceback" not in r.text


def test_only_one_training_job_at_a_time(client):
    first = client.post("/api/training", json={"population": 40, "generations": 30})
    second = client.post("/api/training", json=SMALL)
    assert first.status_code == 202
    assert second.status_code == 409
    assert client.get("/api/training/status").json()["busy"] is True
    run_id = first.json()["run_id"]
    partial = client.get(f"/api/runs/{run_id}").json()
    assert partial["state"] == "running" and partial["evaluation"] is None
    # The baseline is replayable the moment the run exists; "best" is not yet.
    replay = lambda kind: client.post(
        "/api/replay", json={"run_id": run_id, "controllers": [{"kind": kind}]}
    )
    assert replay("baseline").status_code == 200
    assert replay("best").status_code == 404
    assert client.post(f"/api/runs/{run_id}/evaluate").status_code == 409
    client.manager.wait(300)
    assert client.get(f"/api/runs/{run_id}").json()["state"] == "completed"


def test_failed_run_is_recorded_as_failed(client, monkeypatch):
    import creature.jobs as jobs

    def boom(*a, **k):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(jobs, "train", boom)
    run_id = _train(client)
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["state"] == "failed"
    assert run["evaluation"] is None
    assert "secret detail" not in json.dumps(run)


def test_concurrent_reads_never_see_a_partial_write(tmp_path):
    import threading

    store = RunStore(tmp_path)
    run_id = store.create({})
    payload = {"generations": [{"generation": i, "best_fitness": i * 1.5} for i in range(300)]}
    store.write(run_id, "metrics", payload)
    errors: list[Exception] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                assert store.read(run_id, "metrics") == payload
            except Exception as exc:  # noqa: BLE001 - the test reports any failure
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(200):
        store.write(run_id, "metrics", payload)
    stop.set()
    for t in threads:
        t.join()
    assert errors == []
