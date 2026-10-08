"""Local run storage: plain JSON files, written atomically."""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
_FILES = ("run", "metrics", "checkpoints", "evaluation")


class RunNotFound(Exception):
    pass


class RunStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root) / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        # One process owns the store. The lock keeps readers away from a file
        # that is mid-replace, which is not atomic on every bind-mounted filesystem.
        self._lock = threading.RLock()

    def _dir(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise RunNotFound(run_id)
        return self.root / run_id

    def write(self, run_id: str, name: str, data: dict) -> None:
        assert name in _FILES
        path = self._dir(run_id) / f"{name}.json"
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)

    def read(self, run_id: str, name: str) -> dict | None:
        assert name in _FILES
        path = self._dir(run_id) / f"{name}.json"
        with self._lock:
            if not path.exists():
                if name == "run":
                    raise RunNotFound(run_id)
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def create(self, config: dict) -> str:
        now = datetime.now(timezone.utc)
        run_id = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        self._dir(run_id).mkdir()
        self.write(
            run_id,
            "run",
            {"id": run_id, "state": "running", "created": now.isoformat(), "config": config},
        )
        return run_id

    def set_state(self, run_id: str, state: str) -> None:
        with self._lock:
            run = self.read(run_id, "run")
            run["state"] = state
            self.write(run_id, "run", run)

    def list(self) -> list[dict]:
        runs = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if d.is_dir() and RUN_ID.match(d.name) and (d / "run.json").exists():
                runs.append(self.read(d.name, "run"))
        return runs

    def mark_interrupted(self) -> None:
        """Runs left 'running' by a previous process can never finish."""
        for run in self.list():
            if run["state"] == "running":
                self.set_state(run["id"], "interrupted")

    def genome(self, run_id: str, kind: str, generation: int | None = None) -> np.ndarray:
        ck = self.read(run_id, "checkpoints")
        if ck is None:
            raise RunNotFound(run_id)
        if kind == "generation":
            bests = ck["generation_best"]
            if generation is None or not 0 <= generation < len(bests):
                raise KeyError("generation")
            return np.array(bests[generation])
        if ck.get(kind) is None:
            raise KeyError(kind)
        return np.array(ck[kind])
