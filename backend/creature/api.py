"""Local HTTP API and static frontend hosting."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import HELD_OUT_SEEDS, UNSEEN_SEEDS, WORLD, TrainConfig
from .evaluation import default_baseline, evaluate, replay
from .jobs import TrainingManager
from .scenario import VARIANTS, generate
from .storage import RUN_ID, RunNotFound, RunStore

log = logging.getLogger("creature.api")
_REPO = Path(__file__).resolve().parents[2]

Variant = Literal["standard", "dense_hazards", "more_obstacles", "edge_food"]


class ControllerRef(BaseModel):
    kind: Literal["baseline", "best", "generation", "greedy"]
    generation: int | None = Field(None, ge=0, le=300)
    model_config = {"extra": "forbid"}


class ReplayRequest(BaseModel):
    run_id: str | None = Field(None, pattern=RUN_ID.pattern)
    seed: int = Field(HELD_OUT_SEEDS[0], ge=0, le=2**31 - 1)
    variant: Variant = "standard"
    controllers: list[ControllerRef] = Field(min_length=1, max_length=3)
    model_config = {"extra": "forbid"}


def create_app(artifacts_dir: str | os.PathLike | None = None, frontend_dir: str | os.PathLike | None = None) -> FastAPI:
    store = RunStore(artifacts_dir or os.environ.get("ARTIFACTS_DIR", _REPO / "artifacts"))
    store.mark_interrupted()
    manager = TrainingManager(store)
    app = FastAPI(title="The Learning Creature", docs_url=None, redoc_url=None)
    app.state.manager = manager
    app.state.store = store

    @app.exception_handler(RunNotFound)
    async def _not_found(request: Request, exc: RunNotFound):
        return JSONResponse({"detail": "run not found"}, status_code=404)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        return {
            "defaults": TrainConfig().model_dump(),
            "variants": list(VARIANTS),
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "unseen_seeds": list(UNSEEN_SEEDS),
            "max_steps": WORLD.max_steps,
        }

    @app.post("/api/training", status_code=202)
    def start_training(cfg: TrainConfig):
        run_id = manager.start(cfg)
        if run_id is None:
            raise HTTPException(409, "a training run is already in progress")
        return {"run_id": run_id}

    @app.get("/api/training/status")
    def training_status():
        return {"busy": manager.busy(), "run_id": manager.current}

    @app.get("/api/runs")
    def list_runs():
        return {"runs": store.list()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = store.read(run_id, "run")
        metrics = store.read(run_id, "metrics") or {"generations": []}
        return {
            **run,
            "generations": metrics["generations"],
            "evaluation": store.read(run_id, "evaluation"),
        }

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate_run(run_id: str):
        run = store.read(run_id, "run")
        if run["state"] != "completed":
            raise HTTPException(409, "run has not completed training")
        cfg = TrainConfig(**run["config"])
        result = evaluate(cfg, store.genome(run_id, "baseline"), store.genome(run_id, "best"))
        store.write(run_id, "evaluation", result)
        return result

    @app.post("/api/replay")
    def run_replay(req: ReplayRequest):
        controllers = []
        for ref in req.controllers:
            if ref.kind == "greedy":
                controllers.append(("greedy", None))
            elif req.run_id is None:
                if ref.kind != "baseline":
                    raise HTTPException(400, "only the baseline exists without a run")
                controllers.append(("baseline", default_baseline()))
            else:
                try:
                    genome = store.genome(req.run_id, ref.kind, ref.generation)
                except KeyError:
                    raise HTTPException(404, "controller checkpoint not available")
                name = ref.kind if ref.kind != "generation" else f"generation {ref.generation}"
                controllers.append((name, genome))
        return replay(generate(req.seed, req.variant), controllers)

    dist = Path(frontend_dir or os.environ.get("FRONTEND_DIST", _REPO / "frontend" / "dist"))
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
