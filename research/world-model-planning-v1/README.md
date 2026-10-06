# World-model planning pilot

A new local pilot, separate from the closed language-model organ and routing studies.
The user chose world models first on 6 October 2026. This has no language host,
GPT-2 run, provider call or paid inference.

The question: in one small world, how much navigation failure comes from learned
transition errors, and how much remains when the same planner has exact dynamics?

## Fixed experiment

A 5 × 5 map has a wall down its middle, a key at (0,4), and a door at (2,2).
The key is permanent. Four actions move north, east, south and west; invalid
moves leave the state unchanged. Enumerating from (0,0,no key) gives 30 states.
A goal means reaching a position with either key status. All 600 nontrivial
start–goal pairs are reachable.

The direct predictor has 7,198 parameters; the joint state/action/candidate energy
scorer has 7,243. Both use the same public valid-state vocabulary, observations,
96 training transitions, 1,500 Adam updates, 32-example batches, learning rate
0.003 and three fixed seeds (11, 29, 47). Both minimize categorical negative
log likelihood. The energy scorer's categorical logits are **negative energies**.
This compares two parameterizations, not different learning objectives. Evaluating
all candidates makes their computational costs unequal; elapsed time is reported.

Hold out six transitions for each action (24 total), with key pickup and door
entry represented on both sides of the split. No test target enters a loss.
All six fits finish before held-out evaluation or planning. There is no early
stopping, model selection or hyperparameter search.

Planning uses the full predicted transition distribution and exact finite-horizon
expectation. Horizons are 1, 3 and 16. Each step costs one until the goal;
the terminal cost is Manhattan distance. Replan from the true observed state.
At ties within 1e-12, take the first action in N/E/S/W order. Stop at success,
a repeated full state, or 40 actions. Returning to the same position with a key
is not a cycle. The exact-dynamics planner uses identical rules. Full BFS gives
the shortest-path reference. Exact dynamics at horizon 16 must solve all 600.

Primary descriptive outputs are held-out transition accuracy and capped NLL and all 600
navigation outcomes at each horizon. Capped NLL uses float64 tiny as a floor; zero true-probability counts are also
reported (their true NLL is infinite). Separate wrong-action and action-average
controls test action dependence at the transition level. Four fixed ten-action
sequences per state measure open-loop predicted distributions at steps 1,3,6,10.
Planning episodes are closed-loop and must not be described as blind rollouts.

## Scope

This is one-map development, not independent confirmation or unseen-world testing.
The label vocabulary excludes unreachable states by construction; zero invalid
state mass is therefore a support constraint, not a learned capability. Goal
positions and rollouts were not used for model selection. Seeds quantify the
variation among these six fits, not a population significance test. Longer
lookahead costs more and its success is not an energy-model advantage. The
heuristic, ties and topology constrain interpretation of planner failures.

A useful result here can motivate a later world-model organ. It cannot demonstrate
a language model using that organ or establish novelty.

## Run

Use the existing Python environment with NumPy and PyTorch. Preparation tests:

```sh
work/acquisition-representation-venv/bin/python -B -m unittest discover -s research/world-model-planning-v1 -p 'test_*.py' -v
```

A new output directory is mandatory; completed or stopped directories cannot be
overwritten by the runner:

```sh
work/acquisition-representation-venv/bin/python -B research/world-model-planning-v1/launch.py --out outputs/world-model-planning/V1 --receipt work/world-model-planning-v1/WATCHDOG.json
work/acquisition-representation-venv/bin/python -B research/world-model-planning-v1/audit.py --directory outputs/world-model-planning/V1 --launch-receipt work/world-model-planning-v1/WATCHDOG.json
```

The collector freezes source hashes, saves all six checkpoints and logits, and
retains all episodes. The independent auditor implements its own world, planner
and scoring. It checks saved predictions and outcomes; it does not reproduce
training gradients. Bounds are 900 seconds for collection, 2 GiB RSS, 128 MiB
outputs and at least 8 GiB free disk. No old run is opened for scientific inputs.

Motivation: [JEPA-Anything](https://arxiv.org/html/2609.20800v1) proposes testing
predictive components through interventions; [The Planning Limits of Latent
World Models](https://arxiv.org/html/2609.39235v1) motivates exact-dynamics and
horizon controls. This finite symbolic pilot is our adaptation, not a replication
of their large visual models.

`launch.py` monitors the collector from a separate process at 0.5-second intervals,
terminates its process group on the time/RSS bound, and never retries. Collection
also checks peak RSS and storage internally. Each prediction array is saved before
its finiteness check; partial returns remain available if collection stops.
