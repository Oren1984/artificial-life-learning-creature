# Experimental results

**What did we actually prove, and what did we not prove?**

Short answer: evolution reliably turned a random network into one that finds
food in layouts it never trained on. It did not produce a creature better than
a one-line hand-written rule, and it did not produce measurable hazard avoidance.

All numbers on this page come from the files in [`results/`](results/):
[`reference-training.json`](results/reference-training.json),
[`reference-evaluation.json`](results/reference-evaluation.json) and
[`replicates.json`](results/replicates.json).

## Protocol

| | |
|---|---|
| Training layouts | `standard`, seeds 100–107 (8 layouts) |
| Held-out layouts | `standard`, seeds 9000–9029 (30 layouts) |
| Unseen layouts | `dense_hazards`, `more_obstacles`, `edge_food`, seeds 9100–9129 (30 each) |
| Episode | up to 500 steps, one episode per controller per layout |
| Training budget | population 40, 50 generations, training seed 0 |
| Model selection | best genome of the final generation by **training** fitness |
| Uncertainty | paired bootstrap over layouts, 10,000 resamples, 95% interval |

Controllers compared on identical layouts:

- **Trained creature** — the selected genome.
- **Untrained (10 random networks)** — the first 10 genomes of the initial
  population, averaged per layout. This is the main reference.
- **Untrained baseline creature** — the single random network shown in the UI.
- **Random actions** — uniform random accelerations, ignoring observations.
- **Greedy heuristic** — hand-written: accelerate straight at the nearest food.

"Success" means all six pieces of food were collected. The simulation is
deterministic, so each controller–layout pair is one episode; the ± values are
standard deviations across the 30 layouts.

**Verdict rule**, fixed in code before results were read: `improved` only if the
95% interval of *trained minus untrained population* is above zero for both
food and reward on the held-out standard layouts.

## Training curve (reference run)

| Generation | Best fitness | Mean fitness | Best food (of 6) | Mean food |
|---|---|---|---|---|
| 0 | 11.5 | 1.7 | 1.13 | 0.25 |
| 5 | 31.1 | 9.8 | 2.88 | 0.98 |
| 10 | 39.8 | 18.7 | 3.50 | 1.75 |
| 20 | 73.5 | 47.5 | 5.88 | 4.00 |
| 30 | 77.9 | 61.5 | 6.00 | 4.92 |
| 40 | 78.3 | 69.6 | 6.00 | 5.46 |
| 50 | 78.7 | 74.0 | 6.00 | 5.74 |

Best fitness cannot decrease because the top two genomes are carried over
unchanged and the training layouts are fixed. This curve shows the optimizer
working; it is not the evidence of learning. The held-out results are.

## Held-out results (in distribution)

| Controller | Food (of 6) | Success | Survival (steps) | Collisions | Hazard steps | Reward |
|---|---|---|---|---|---|---|
| Untrained (10 random networks) | 0.26 ± 0.53 | 0% | 225 | 2.36 | 2.3 | 1.6 |
| Untrained baseline creature | 0.23 ± 0.42 | 0% | 258 | 2.43 | 1.8 | 1.9 |
| Random actions | 0.17 ± 0.45 | 0% | 215 | 2.73 | 2.6 | 0.6 |
| **Trained creature** | **5.57 ± 1.31** | **87%** | **463** | **1.80** | **3.9** | **71.5** |
| Greedy heuristic | 5.77 ± 0.96 | 93% | 471 | 1.60 | 4.4 | 74.7 |

Paired differences (trained minus reference, per layout):

| Comparison | Food | 95% CI | Reward | 95% CI |
|---|---|---|---|---|
| vs. untrained population | +5.31 | 4.79 to 5.69 | +69.9 | 62.7 to 75.2 |
| vs. untrained baseline | +5.33 | 4.80 to 5.73 | +69.6 | 62.6 to 75.1 |
| vs. greedy heuristic | −0.20 | −0.83 to 0.40 | −3.2 | −11.8 to 5.1 |

**Verdict: `improved`.** The trained creature collected more food than the
untrained population on 29 of 30 layouts. On the remaining one (seed 9008) it
collected none. Its per-layout food counts were 6 on 26 layouts and 5, 4, 2
and 0 on the other four.

## Unseen scenarios (no retraining)

| Variant | Trained food | Trained success | Greedy food | Greedy success | Trained − untrained food (95% CI) | Trained − greedy food (95% CI) |
|---|---|---|---|---|---|---|
| Dense hazards | 5.37 ± 1.33 | 77% | 5.63 | 80% | +5.03 (4.55 to 5.44) | −0.27 (−0.87 to 0.30) |
| More, larger obstacles | 5.83 ± 0.73 | 93% | 5.97 | 97% | +5.52 (5.23 to 5.70) | −0.13 (−0.40 to 0.00) |
| Food along the walls | 5.63 ± 0.95 | 83% | 6.00 | 100% | +5.33 (4.96 to 5.62) | −0.37 (−0.73 to −0.07) |

Untrained networks score 0.31–0.33 food and 0% success on all three variants.

Hazard steps per episode on the dense-hazard variant: trained 7.3, greedy 9.5.

## Run-to-run variation

Five independent training runs (training seeds 0–4), same protocol. All five
received the verdict `improved`.

| Training seed | Held-out food | Held-out success | Dense hazards | More obstacles | Edge food |
|---|---|---|---|---|---|
| 0 (reference) | 5.57 | 87% | 5.37 / 77% | 5.83 / 93% | 5.63 / 83% |
| 1 | 5.73 | 93% | 5.60 / 83% | 5.97 / 97% | 5.23 / 70% |
| 2 | 5.57 | 83% | 5.70 / 83% | 6.00 / 100% | 5.93 / 97% |
| 3 | 5.60 | 80% | 4.13 / 37% | 5.50 / 77% | 4.90 / 53% |
| 4 | 4.93 | 73% | 4.30 / 43% | 5.07 / 70% | 5.40 / 73% |

Unseen columns are trained food / success. The greedy heuristic scores
5.77 / 93% on the held-out layouts regardless of training seed, so no run beat
it on food, and seed 1 matched its success rate.

The reference run is seed 0 because it is the default configuration and was
run first, not because it was the best: seeds 1 and 2 are as good or better,
seeds 3 and 4 are worse.

## What was proven

1. **Genuine optimization took place.** Controller weights changed, and the
   change is the only difference between the baseline and trained creature.
2. **Food-seeking was learned and holds on held-out layouts.** +5.31 food with
   an interval far from zero, in all five training runs.
3. **The behavior transfers to world variations never trained on**, in the
   sense of remaining far above untrained (intervals above +4.5 food in every
   variant of the reference run).
4. **Results are reproducible** from the configuration and seeds.

## What was not proven

1. **That the creature learned anything a trivial rule does not already do.**
   It is statistically indistinguishable from the greedy heuristic on held-out
   layouts, slightly behind on every point estimate, and measurably worse on
   the edge-food variant. The honest reading is that evolution rediscovered
   "go to the nearest food" — which is still learned, since nothing told the
   network to use inputs 2–3 that way, but it is a modest result.
2. **Hazard avoidance.** Trained hazard steps (3.9) are higher than the
   untrained population's (2.3) — plausibly because the trained creature survives twice
   as long and covers far more ground — and only slightly below greedy's (4.4).
   No paired interval was computed for hazard steps, so even the small gap to
   greedy is a point estimate, not a finding. The mission's question included
   "avoid hazards"; this experiment does not support a yes.
3. **Obstacle avoidance.** Collisions per episode fall from 2.36 to 1.80, but
   per 100 steps survived they are flat (1.05 untrained, 1.09 trained, 1.11
   greedy).
4. **Robust generalization.** Two of five training runs dropped to 37% and 43%
   success on dense hazards.
5. **Anything about other algorithms.** No comparison with NEAT, policy
   gradients or Q-learning was made.

## Tuning history

Two full training runs were made in total. This is the complete list.

1. **First run** — hazard energy drain 0.03 per step, hazard reward −0.05.
   Trained: 5.67 food, 90% success. Greedy heuristic: 6.00 food, 100% success.
   A rule that ignores hazards scored perfectly, which meant hazards had no
   real effect on the task.
2. **Second run (current)** — hazard drain raised to 0.08 and hazard reward to
   −0.2. No further changes.

This was a change to the world's rules, applied equally to every controller,
made before any result was reported. It was also made after seeing held-out
numbers from the first run, so the held-out set is not perfectly untouched:
it informed one design decision. Note that the change made the trained
creature's headline numbers slightly *lower* (5.67 → 5.57 food), not higher.
The evaluation protocol, seeds and verdict rule were not changed.

## Reproducing

```bash
cd backend
python -m creature.cli train --seed 0                      # reference run
python -m creature.cli replicates --seeds 0 1 2 3 4 --out ../docs/results/replicates.json
```

Or start a run from the UI with the default settings and read the tables in
the *Before vs. after* view.
