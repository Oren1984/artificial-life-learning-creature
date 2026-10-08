# Research foundation

This project borrows a few well-established principles from three references.
It does not reproduce any of them. For each one, this page separates what the
source investigates, what this POC implements, what the experiment shows, and
what is out of scope.

## Reference A — Reinforcement learning

Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*,
2nd edition. <https://incompleteideas.net/book/the-book-2nd.html>

- **What the source investigates.** Learning from interaction: an agent
  observes a state, acts, receives a reward, and adjusts its policy to maximize
  long-run reward. It covers value functions, temporal-difference learning,
  policy gradients, and the exploration–exploitation trade-off.
- **What this POC implements.** The agent–environment interaction loop; an
  explicit observation space, action space and reward; a policy as a function
  from observations to actions; episodic tasks with defined termination; and
  the practice of judging a policy by separate evaluation rather than by its
  training score.
- **What the experiment demonstrates.** That a policy optimized against an
  episodic reward improves on held-out episodes of the same task.
- **Out of scope.** No value function, no temporal-difference update, no policy
  gradient, no per-step credit assignment. The reward is only used as an
  episode-level fitness score, so this is *not* a reinforcement-learning
  algorithm in the book's sense. Exploration comes only from mutation noise.

## Reference B — Neuroevolution

Kenneth O. Stanley and Risto Miikkulainen, *Evolving Neural Networks through
Augmenting Topologies*, Evolutionary Computation 10(2), 2002.
<https://doi.org/10.1162/106365602320169811>

- **What the source investigates.** NEAT: evolving both the weights *and the
  structure* of neural networks, using historical markings to align genomes for
  crossover, speciation to protect new structures, and growth from minimal
  topologies.
- **What this POC implements.** The general neuroevolution idea the paper
  builds on: a population of neural controllers, fitness measured by running
  them in an environment, selection, crossover and mutation, and improvement
  across generations.
- **What the experiment demonstrates.** That plain weight evolution on a fixed
  14→12→2 network is enough to learn food-seeking in this world.
- **Out of scope.** Everything that makes NEAT NEAT: topology mutation,
  historical markings, speciation, complexification. **This project does not
  implement or reproduce NEAT** and makes no claim about its results.

## Reference C — Environments

Farama Foundation, Gymnasium. <https://gymnasium.farama.org/>

- **What the source provides.** A standard interface for reinforcement-learning
  environments: `reset`, `step`, observation and action spaces, termination and
  truncation, and seeding for reproducibility.
- **What this POC implements.** The same contract in its own small engine:
  seeded deterministic resets, a step function, bounded normalized
  observations, bounded actions, explicit termination (all food eaten, energy
  exhausted) and truncation (500 steps).
- **What the experiment demonstrates.** That the whole pipeline is reproducible
  from seeds: the same configuration yields the same training history and the
  same evaluation numbers.
- **Out of scope.** The Gymnasium library is not a dependency and the engine
  does not subclass `gymnasium.Env`. The engine is batched (many creatures per
  step) for speed, which the single-environment Gymnasium API does not describe.

## How the principles shaped the design

| Principle | Design decision |
|---|---|
| Agent–environment loop (A, C) | One `World` class owns sensing, physics and reward; controllers only see observations. |
| Evaluation of learned behavior (A) | Held-out seeds that training can never use, plus unseen world variants. |
| Population-based optimization (B) | Genetic algorithm with elitism, tournament selection, uniform crossover, Gaussian mutation. |
| Seeded reproducibility (C) | A layout is fully defined by `(seed, variant)`; training by one integer seed. |

## Limitations of this page

- These are summaries of what the project adopted, not literature reviews.
- Link check from the development machine on 2026-10-08: the Gymnasium site
  returned HTTP 200. The DOI link returned 403 to a command-line client (the
  publisher blocks non-browser requests) and incompleteideas.net timed out, so
  those two URLs were **not** confirmed reachable from this machine. They are
  the URLs given in the project brief.
