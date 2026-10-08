# Testing

Three suites. Results below are from the last run on 2026-10-08 (Windows 11,
Python 3.12.10, Node 24, Docker 29.8).

| Suite | Command | Result |
|---|---|---|
| Backend (pytest) | `cd backend && python -m pytest -q` | 52 passed |
| Frontend unit (Vitest, jsdom) | `cd frontend && npx vitest run` | 25 passed |
| Browser E2E (Playwright) | `cd frontend && npx playwright test e2e/app.spec.ts` | 12 passed (6 tests × desktop and mobile) |

Setup: `pip install -r backend/requirements-dev.txt` and `npm install` in
`frontend/`. The E2E suite needs the app running (`docker compose up`) and
Google Chrome installed.

## Backend

`tests/test_world.py` — the simulation engine

- Scenario generation is deterministic and well-formed for all four variants.
- Reset is deterministic; observations stay within bounds.
- One step matches the physics computed by hand; actions are clipped.
- Wall and obstacle collisions block movement and are counted once per contact.
- Food collection restores energy; hazards drain it.
- Episodes end by starvation, by success, and never exceed the time limit.
- Reward equals the sum of its measured terms.
- Progress reward cannot be farmed by oscillating.
- A batched simulation matches the same episodes run one at a time.

`tests/test_learning.py` — learning integrity

- Controller inference: shape, bounds, determinism; wrong genome size rejected.
- Mutation changes children but preserves elites; selection favours fitter parents.
- Training is reproducible from its seed and actually changes parameters.
- Reported fitness is recomputed from simulation and matches.
- Generation 0 is the unmodified random population.
- Elitism never loses training fitness.
- Evaluation seeds are disjoint from training seeds, and a configuration that
  overlaps them is rejected.
- Evaluation uses the controllers it was given.
- **A failed learning experiment is not reported as success**, and the verdict
  rule returns `improved`, `no_clear_evidence` and `regressed` correctly.
- Replay matches the simulation and is deterministic.

`tests/test_storage_api.py` — storage and API

- A saved checkpoint reproduces the saved policy exactly.
- Invalid run ids are rejected; runs left running are marked interrupted.
- Full training-run lifecycle through the API; replay with and without a run.
- Validation errors, missing runs and missing checkpoints return clean errors.
- Error responses do not leak file paths.
- Only one training job runs at a time; a crashed job is recorded as failed.
- Concurrent readers never see a partially written file.

No test depends on training producing an improvement.

## Frontend unit

- Initialization: untrained creature with no experiment; a completed
  experiment loads as a replay, not as live training; an unreachable backend is
  reported.
- Training progress: start, poll real metrics, finish; rejected request; failed
  run; generation slider replays the chosen generation.
- Before/after: both controllers requested for one identical scenario; a weak
  result is shown as weak; nothing offered before training completes.
- Controls: pause, resume, reset on the shared clock; scenario change reloads
  the replay; replay errors recover.
- Chart mapping, tick generation, replay-clock interpolation, evaluation table rows.
- Markup escaping, mobile viewport, single-column layout on small screens.

## Browser E2E

Run against the Docker container in real Chrome, at 1280×860 and at a 390×844
touch viewport:

1. Page loads and the untrained creature renders on the canvas.
2. Pause, resume and reset control the replay clock.
3. Training runs live (population 12, 6 generations), then baseline and trained
   are compared on the same scenario.
4. Switching scenario loads a different layout.
5. A completed experiment survives a page reload.
6. The layout fits the viewport without horizontal scrolling.

Every test fails on any uncaught page error, console error, failed request or
HTTP status ≥ 400.

`e2e/screenshots.spec.ts` is not a test. It captures the images in
`docs/images/` and only runs with `SCREENSHOTS=1`.

## Docker and integration checks (manual)

- `docker compose up --build` builds and starts; the healthcheck reports healthy.
- Container logs show no tracebacks or 5xx responses after the E2E run.
- Runs persist across `docker compose restart` and across `down` / `up`.
- Re-evaluating the reference run after a restart reproduced identical results.
- Native start with `uvicorn` served the API and the built frontend.

## Defects found by testing, and fixed

1. Replaying the baseline right after starting a run returned 404, because the
   checkpoint was written by the worker thread slightly later. The baseline
   checkpoint is now written when the run is created.
2. Reading a JSON file while it was being replaced on the Windows bind mount
   occasionally raised `OSError`. Store reads and writes now share a lock; a
   regression test covers it.

## Known gaps

- **One unreproduced E2E failure.** In one run started immediately after an
  image rebuild, the first desktop test timed out waiting 5 s for the initial
  page to leave its "Loading…" state. The other 11 tests in that run passed,
  the backend returned 200 for every request, and a full re-run after
  recreating the container passed 12 of 12. Cold-start request timings measured
  afterwards were all under 0.2 s. The cause was not established.
- E2E uses the locally installed Chrome (`channel: "chrome"`); Playwright's
  bundled browsers were not downloaded, so Firefox and WebKit are untested.
- "Mobile" is an emulated viewport in desktop Chrome, not a real device.
- The E2E suite stores small training runs in whatever instance it targets.
  Point `BASE_URL` at a throwaway container (no volume) to keep `artifacts/` clean.
- The Docker checks above were performed by hand, not by an automated script.
- There is no CI configuration.
