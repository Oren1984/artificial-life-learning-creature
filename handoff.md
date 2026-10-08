# Handoff — Artificial Life: The Learning Creature

Updated 2026-10-08 (final closure). **The project is closed with the verdict
PASS WITH LIMITATIONS.** Nothing remains to be done.

## State of the repository

- Branch `main`. The whole delivery is in one local commit on top of
  `cab4481 Initial commit`. **Nothing has been pushed or deployed.**
- The Docker container may still be running on `http://localhost:8080`
  (`docker compose down` to stop it).
- `artifacts/` (gitignored) holds only the reference run
  `20261008-170954-c1d7bb` (default config, seed 0) and `replicates.json`.
  The 12 throwaway E2E runs (population 12, 6 generations) were deleted after
  checking that nothing referenced them.

## Done at closure

- Wrote `README.md` (10 sections) and the six `docs/*.md` files; all relative
  links resolve.
- Made the 77-second demo video: `docs/media/learning-creature-demo.mp4`, with
  poster, captions, recording facts and the scripts that produced it in
  `docs/media/source/`. See `docs/media/README.md`. The run recorded for the
  video reproduced the reference metrics and evaluation exactly.
- Final checks: backend 52 passed, frontend unit 25 passed, browser E2E 12
  passed (run against a throwaway container so no new runs were stored).
- One E2E timeout seen once earlier and never reproduced is disclosed in
  `docs/testing.md`.

## Notes for anyone picking this up later

- Running the E2E suite against the compose container stores four small runs
  per pass in `artifacts/runs/`. To avoid that, start a throwaway instance:
  `docker run -d --rm -p 127.0.0.1:8092:8080 learning-creature:local` and run
  with `BASE_URL=http://localhost:8092`.
- A bind mount under the Windows Temp directory made the backend several times
  slower; use the repository's `./artifacts` or no volume.
- No further research, tuning or scope expansion is planned.

## Completed (first session)

**Backend** (`backend/creature/`, Python + NumPy + FastAPI)
- `config.py` — frozen world/reward constants, validated `TrainConfig`, seed sets
  (train 100–107, held-out 9000–9029, unseen 9100–9129; overlap is rejected).
- `scenario.py` — deterministic layouts; variants `standard`, `dense_hazards`,
  `more_obstacles`, `edge_food`.
- `world.py` — the single batched simulation engine (14 observations, 2 actions)
  used by training, evaluation and replay.
- `policy.py` — fixed-topology MLP 14→12→2 (206 weights), greedy heuristic,
  random-action policy.
- `evolution.py` — genetic algorithm (elitism, tournament, uniform crossover,
  Gaussian mutation). Not NEAT.
- `evaluation.py` — held-out + unseen-variant evaluation, paired bootstrap 95% CI,
  `learning_verdict`, step-by-step `replay`.
- `storage.py`, `jobs.py`, `api.py`, `main.py`, `cli.py` — JSON run store, single
  background training job, HTTP API, CLI (`train`, `replicates`).

**Frontend** (`frontend/`, TypeScript + Vite + Canvas 2D)
- Three views: Initial creature, Learning progress (live chart, generation slider),
  Before vs. after (synchronized panes + measured evidence tables).
- Replays only backend trajectories; it does not simulate anything itself.

**Infrastructure** — `Dockerfile` (multi-stage, non-root, healthcheck),
`docker-compose.yml` (127.0.0.1:8080, `./artifacts` mounted), `.gitignore`,
`.dockerignore`.

**Tests — last run, all passing**
- Backend: `cd backend && python -m pytest -q` → 52 passed.
- Frontend unit: `cd frontend && npx vitest run` → 25 passed.
- Browser E2E vs. the Docker container, system Chrome, desktop + mobile:
  `cd frontend && npx playwright test e2e/app.spec.ts` → 12 passed.
- Container logs: no tracebacks or 5xx. Runs persist across `docker compose restart`,
  and re-evaluating the reference run after restart reproduced identical results.
- Native start (`python -m uvicorn creature.main:app --port 8090` from `backend/`)
  served the API and the built frontend.

**Defects found and fixed during QA**
1. Baseline replay returned 404 right after training started (checkpoint written
   by the worker thread too late) → baseline checkpoint now written at run creation.
2. Transient `OSError: No data available` reading a JSON file mid-replace on the
   Windows bind mount → store reads/writes serialized with a lock; regression test added.

**Experiment results** (exported to `docs/results/`)
- `reference-training.json`, `reference-evaluation.json`, `replicates.json`.
- Reference run, 30 held-out standard layouts, food collected out of 6:
  untrained population 0.26 (0% success) · trained 5.57 (87%) ·
  greedy heuristic 5.77 (93%) · random actions 0.17.
- Trained minus untrained food: +5.31, 95% CI 4.79 to 5.69 → verdict `improved`.
- Trained vs. greedy: −0.20, CI −0.83 to 0.40 → not distinguishable; on `edge_food`
  the trained creature is measurably worse than greedy (−0.37, CI −0.73 to −0.07).
- Unseen variants (trained): dense_hazards 5.37 / 77%, more_obstacles 5.83 / 93%,
  edge_food 5.63 / 83%.
- Five training seeds (0–4), held-out food: 5.57, 5.73, 5.57, 5.60, 4.93; all
  `improved`. Unseen results vary a lot by seed (dense_hazards success 37%–83%).
- No evidence the creature learned hazard avoidance: trained hazard steps per
  episode (3.9) are not lower than greedy (4.4) in any convincing way, and one
  held-out layout (seed 9008) scores 0 food.

**Tuning history to disclose** (two runs total)
1. First run with hazard drain 0.03 and hazard reward −0.05: trained 5.67 food / 90%,
   but greedy scored 6.00 / 100%, so hazards were irrelevant.
2. Hazard drain raised to 0.08 and hazard reward to −0.2 (current values). This was a
   world-design change made before any result was reported; no further tuning.

**Screenshots** (`docs/images/`, captured from the running app by
`frontend/e2e/screenshots.spec.ts`): `01-initial-creature`, `02-training-live`,
`03-training-complete`, `04-before-after`, `05-measured-evidence`,
`06-mobile-before-after`. All six were reviewed.

## Known limitations

See `docs/limitations.md` and `docs/testing.md` (Known gaps).
