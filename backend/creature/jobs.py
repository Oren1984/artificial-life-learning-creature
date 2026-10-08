"""Training jobs: one background thread at a time, progress persisted every generation."""
from __future__ import annotations

import logging
import threading

from .config import TrainConfig
from .evaluation import evaluate
from .evolution import BASELINE_INDEX, initial_population, train
from .storage import RunStore

log = logging.getLogger("creature.jobs")


def start_run(store: RunStore, cfg: TrainConfig) -> str:
    """Create a run with its baseline checkpoint, so it is replayable immediately."""
    run_id = store.create(cfg.model_dump())
    baseline = initial_population(cfg)[BASELINE_INDEX].tolist()
    store.write(run_id, "checkpoints", {"baseline": baseline, "best": None, "generation_best": []})
    store.write(run_id, "metrics", {"generations": []})
    return run_id


def run_training(store: RunStore, run_id: str, cfg: TrainConfig) -> None:
    """Train, checkpoint and evaluate a started run. Used by both the API and the CLI."""
    history: list[dict] = []
    bests: list[list[float]] = []
    baseline = store.read(run_id, "checkpoints")["baseline"]

    def checkpoint(best=None):
        # Checkpoints first: a generation listed in metrics is always replayable.
        store.write(
            run_id,
            "checkpoints",
            {"baseline": baseline, "best": best, "generation_best": bests},
        )
        store.write(run_id, "metrics", {"generations": history})

    def on_generation(record: dict, genome) -> None:
        history.append(record)
        bests.append(genome.tolist())
        checkpoint()

    result = train(cfg, on_generation)
    checkpoint(best=result["best"].tolist())
    store.write(run_id, "evaluation", evaluate(cfg, result["baseline"], result["best"]))
    store.set_state(run_id, "completed")


class TrainingManager:
    def __init__(self, store: RunStore):
        self.store = store
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.current: str | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: TrainConfig) -> str | None:
        """Start a run; returns None if one is already in progress."""
        with self._lock:
            if self.busy():
                return None
            run_id = start_run(self.store, cfg)
            self.current = run_id
            self._thread = threading.Thread(target=self._work, args=(run_id, cfg), daemon=True)
            self._thread.start()
            return run_id

    def _work(self, run_id: str, cfg: TrainConfig) -> None:
        try:
            run_training(self.store, run_id, cfg)
        except Exception:
            log.exception("training run %s failed", run_id)
            self.store.set_state(run_id, "failed")

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)
