# Can four useful observations improve a world model?

Accepted protocol, 6 October 2026. New local study; previous studies are immutable.

## Question and fixed panel

Compare random, predictive-entropy and decision-sensitivity selection of four
resettable transition queries. Resetting is free; one observed state/action
successor costs one query. A matched-update replay arm observes nothing new.

Enumerate the 44 base 5x5 maps: wall at x=2 except door (2,d), d=0..4;
key (x,y) with x=0..1,y=0..4, excluding (0,0) and the old (0,4),d=2 layout.
Start (0,0,no key). Sort by SHA-256 of
`wm-active-v1:20261006:{door_y}:{key_x}:{key_y}` and take 12. Rotate map i
clockwise by i modulo 4 quarter-turns, (x,y)->(4-y,x), including its start.
Maps 0..3 are engineering; maps 4..11 are untouched final evaluation.
Each has 30 reachable states, 120 transitions and 600 nontrivial start/goal pairs.
State vocabulary and goal coordinates are public. Door/key/wall labels and true
transitions are unavailable to selectors; public support can reveal geometry.

Within each action, sort row IDs (state_index*4+action) by SHA-256 of
`wm-active-v1:split:{map_id}:{row_id}`. First18 train, next6 query, last6 audit:
72/24/24. No forced events, map replacement or failure-based filtering.

## Learners and acquisition

Use the pilot's direct16->64->64->30 predictor (7,198 parameters) and joint
28->71->71->1 energy scorer (7,243). GELU, categorical CE on negative energies,
Adam lr0.003, no weight decay, float32, deterministic CPU, two threads.
Seeds11,29,47; initial1500 updates, batch32 sampled with replacement using
NumPy default_rng(seed), shared between architectures on a map.
Models are freshly trained per map; this is not zero-shot map transfer.

All initial fits finish before acquisition. Each arm copies the identical model
and Adam state. Four sequential distinct queries; re-score after each learning
update. Save rounds0,1,2,3,4; report0,1,2,4. After each query train250 steps with
16 original examples and16 sampled from all acquired observations. Replay uses
32 original examples. RNG seed+100000*round first samples original slots, then
uniform[0,1) for buffer indices; this pairs sampling streams across arms/models.
Every batch and target hash is retained. No exact-row injection into predictions.

Random ranks by SHA-256 of `wm-active-v1:random:{map_id}:{seed}:{row_id}`.
Entropy is -sum(P log P), zero mass contributes zero. All score ties within1e-12
use SHA-256 of `wm-active-v1:tie:{map_id}:{seed}:{row_id}`. Seeds are replicates,
not a committee. Selectors accept only P, states, goals, candidate IDs, method,
map ID and seed. They cannot call the world, access targets or grade outcomes.

Decision sensitivity uses V15 from the fixed DP and Q(s,a)=1+sum(P V15).
For each goal, I(s,a)=min_b Q(s,b)-sum_t P(t|s,a)*min(min_b!=a Q(s,b),1+V15(t)).
Weight by predicted visits under the current stationary H16 policy over40 steps.
Initial mass is1/600 per non-goal state for each goal; remove all mass reaching
the goal. Accumulate visits before each transition; no pathwise cycle stopping
in this occupancy calculation. Raw gain below-1e-12 stops the study; negative
roundoff within tolerance is logged and clamped to zero. This heuristic assumes
exact row revelation and freezes downstream values. Actual learning changes many
rows, and confidently wrong predictions can get zero entropy and zero sensitivity.

## Planner, phases and endpoints

H16 expected unit-step-cost DP, terminal Manhattan distance, goal value0 each
iteration; first N/E/S/W action within1e-12 of minimum. Replan from true state.
Stop on goal before repeated full-state cycle, or40 actions. Exact dynamics with
the identical planner must solve all600 at BFS-optimal length. Graph-only
preflight is allowed before training; never use oracle paths in acquisition.

Engineering phase completes all acquisition and saves ACQUISITION_COMPLETE before
true navigation or audit-target grading. Independent audit must pass before the
evaluation phase starts. Code, manifest and splits are frozen before fits, with
source snapshots. The evaluation freeze binds the engineering audit. No outcome-
driven tuning, seed/architecture/horizon choice or checkpoint selection.

Primary: per evaluation map, decision-minus-random H16 success fraction at four
queries, equally averaged over two architectures and three seeds; then equal map
mean. Report all eight differences, both architectures and all seeds. No episode-
level significance test or population claim. Secondary: budgets1/2, uncertainty,
zero-query matched-update replay, audit24 accuracy/capped NLL/true probability,
navigation steps/cycles, query overlaps and selection/training time. Capped NLL
floors at float64 tiny; exact zeros are separately counted (true NLL infinite).

Independent code reconstructs maps, splits, selection, batches, prediction
metrics and all600 saved episode traces per condition. It verifies hashes and
phase barriers; it does not replay optimization gradients. Equal labels/updates
and near-equal parameter counts are not FLOP matching.

## Separate earlier-predictor diagnostic

Use only saved direct_11 predictions from the original pilot. Census prediction
errors on versus off executed state/action rows; policy-weighted proper losses;
chosen action membership in the BFS-optimal action set. Cross soft/argmax-only
predictions with true-state feedback/blind internal-state execution. Blind state
advances by lowest-index successor argmax; external success uses the true goal.
Stop at success, repeated (true,internal) pair or40 steps. Internal-goal actions
remain those of the original fixed policy, even when the true goal is not reached.
Retain all600 cases in each cell. Also replace each of24 held-out rows separately
and all jointly with exact distributions, using native feedback. These are post-
hoc diagnostic interventions, not acquired learning or unique mechanism proof.

## Resources and artifacts

New outputs only: outputs/world-model-planning/ACQUISITION_V1; diagnostic in its
own fresh directory. No overwriting or automatic retry. External watchdog per
phase:1800sec,2GiB RSS; study outputs<=1GiB, free storage>=8GiB. Compress episode
logs; preserve STOP/partial outputs. $0 provider calls/spending. Prior inclusive
Cloudflare total/reserve7.6774551949 under cap20 remains unchanged; GPT-2 paused.
Publish independently checked descriptive findings, including negative results,
with plain-language page, per-map bundles, source/audit data and hashes. Do not
claim novelty, a JEPA model, a language-host splice or general energy superiority.

Run with work/acquisition-representation-venv/bin/python -B:

```
-m unittest discover -s research/world-model-acquisition-v1 -p 'test_*.py' -v
research/world-model-acquisition-v1/run.py prepare --out outputs/world-model-planning/ACQUISITION_V1
research/world-model-acquisition-v1/launch.py --out outputs/world-model-planning/ACQUISITION_V1 --phase engineering
research/world-model-acquisition-v1/audit.py --out outputs/world-model-planning/ACQUISITION_V1 --phase engineering
research/world-model-acquisition-v1/launch.py --out outputs/world-model-planning/ACQUISITION_V1 --phase evaluation
research/world-model-acquisition-v1/audit.py --out outputs/world-model-planning/ACQUISITION_V1 --phase evaluation
```
