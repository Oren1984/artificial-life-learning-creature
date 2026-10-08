# Limitations

Grouped by kind. Numbers are in [experimental-results.md](experimental-results.md).

## Scientific limitations

- **The task is easy for a trivial rule.** The creature senses the direction of
  the nearest food directly. "Accelerate that way" collects 5.77 of 6 pieces on
  held-out layouts. The trained network matches this and does not exceed it, so
  the experiment shows that evolution can *find* this behavior, not that it can
  find something a person would not write in one line.
- **Hazard avoidance is not demonstrated.** After hazards were made costly
  enough to matter, the trained creature still spends about as long in them as
  the greedy rule does.
- **Obstacle avoidance is not demonstrated.** Collisions per step survived
  are unchanged by training.
- **Reward shaping points at the solution.** The progress term rewards closing
  distance to the nearest food. It does not encode a route, but it makes
  food-seeking the easiest behavior to discover and may be why nothing subtler
  emerged.
- **Generalization depends on the training seed.** Dense-hazard success ranged
  from 37% to 83% across five runs.
- **Small samples.** 30 layouts per suite and 5 training seeds. Intervals
  describe variation across layouts for one trained network, not across
  training runs.
- **The held-out set informed one design decision** (the hazard-cost change
  described in the tuning history).
- **The training curve is not evidence.** It is monotone by construction.
- **Unseen variants are mild.** They change counts, sizes and placement within
  the same world rules; they are not a different task.
- **The trained creature still fails outright** on at least one held-out layout.

## Engineering limitations

- One training job at a time, in a thread of the web process. A restart during
  training loses the job (the run is marked `interrupted`).
- Storage is JSON files with an in-process lock. It assumes a single process.
- No authentication. The compose file binds to `127.0.0.1` only; do not expose
  the port.
- The container runs as uid 1000. On a Linux host `./artifacts` must be
  writable by that user.
- The frontend polls every 500 ms; there is no push channel.
- Replays are computed on request, not cached.
- `npm install` on the development machine blocked esbuild's postinstall
  script; build and tests worked regardless. The Docker build uses
  `npm ci --ignore-scripts`.
- Testing gaps are listed in [testing.md](testing.md#known-gaps).

## Known defects

None open. One intermittent E2E timeout was observed once and not reproduced;
see [testing.md](testing.md#known-gaps).

## Possible next steps

Not planned work — just the most informative directions.

1. **Make avoidance necessary.** Place hazards and obstacles between the
   creature and the food so that the greedy route fails, then see whether
   evolution finds a detour.
2. **Remove the shortcut.** Replace the food-direction input with range
   sensors, and drop or reduce the progress term.
3. **Measure what is claimed.** Add paired intervals for hazard steps and
   collisions per step.
4. **More seeds.** Report results across 20+ training seeds rather than one
   reference run and four replicates.
5. **Try the real thing.** Compare with an actual NEAT implementation or a
   simple policy-gradient method on the same world.
