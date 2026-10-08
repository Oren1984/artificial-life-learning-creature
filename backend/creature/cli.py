"""Command line: run experiments without the browser.

    python -m creature.cli train --seed 0
    python -m creature.cli replicates --seeds 0 1 2 3 4 --out ../docs/results/replicates.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainConfig
from .evaluation import evaluate
from .evolution import train
from .jobs import run_training, start_run
from .storage import RunStore


def _row(suite: dict, name: str) -> dict:
    c = suite["controllers"][name]
    return {k: round(c[k], 3) for k in ("food_mean", "success_rate", "survival_mean", "reward_mean")}


def cmd_train(args) -> None:
    cfg = TrainConfig(seed=args.seed, generations=args.generations, population=args.population)
    store = RunStore(args.artifacts)
    run_id = start_run(store, cfg)
    run_training(store, run_id, cfg)
    ev = store.read(run_id, "evaluation")
    print(f"run {run_id} -> {ev['verdict']['outcome']}")
    for name in ev["in_distribution"]["controllers"]:
        print(f"  {name:22s} {_row(ev['in_distribution'], name)}")


def cmd_replicates(args) -> None:
    """Repeat training with several evolution seeds to show run-to-run variation."""
    rows = []
    for seed in args.seeds:
        cfg = TrainConfig(seed=seed)
        result = train(cfg)
        ev = evaluate(cfg, result["baseline"], result["best"])
        row = {
            "training_seed": seed,
            "final_best_train_fitness": round(result["history"][-1]["best_fitness"], 3),
            "verdict": ev["verdict"]["outcome"],
            "held_out": {n: _row(ev["in_distribution"], n) for n in ("untrained_population", "trained", "greedy_heuristic")},
            "unseen_trained": {v: _row(s, "trained") for v, s in ev["unseen"].items()},
        }
        rows.append(row)
        print(json.dumps(row))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"replicates": rows}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="creature")
    sub = parser.add_subparsers(required=True)
    t = sub.add_parser("train", help="train, checkpoint and evaluate one run")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--generations", type=int, default=TrainConfig().generations)
    t.add_argument("--population", type=int, default=TrainConfig().population)
    t.add_argument("--artifacts", default=str(Path(__file__).resolve().parents[2] / "artifacts"))
    t.set_defaults(func=cmd_train)
    r = sub.add_parser("replicates", help="train with several seeds and summarize")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_replicates)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
