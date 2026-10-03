# Experiment 4: causal visibility as a binding constraint

## Prior failure and new question

Experiment 3's strong position-label sufficiency claim failed in every model.
Those public results are discovery evidence for this new experiment, not
reinterpreted as a positive primary result. Canonical position labels restored
only 29–59% accuracy when all keys preceded all values.

Grouping changes which input nodes each value can attend to. In the training
layout `K0 V0 K1 V1 K2 V2 K3 V3 Q`, Vi can see only K0 through Ki. With grouped
tokens `K0 K1 K2 K3 V0 V1 V2 V3 Q` and canonical position IDs, it can see every
key. Does removing just the extra key edges restore final retrieval, and does
the identity of those edges matter beyond generic sparsification?

All six checkpoints remain unchanged. All execution is local CPU. There is no
new training or E4 trained-model discovery. The E4 predictor may use the now
published E3 outcomes; it cannot inspect E4 inputs or outcomes. Blinding is
procedural in a shared workspace. Nothing in E1, E2 or E3's frozen records changes.

## Exact intervention and controls

Every grouped condition keeps canonical input coordinates `[0,2,4,6,1,3,5,7,8]`
and the ordinary physical causal mask. For every L1 attention head, the correct
guard removes Vi-to-Kj attention edges when j>i, then normalizes over remaining
legal predecessors. Scores are recomputed from cached pre-intervention q/k and
masked before softmax to avoid instability from dividing very small probabilities.
Only the four L1 value query rows are patched; key and final-query rows remain
endogenous and unchanged by this operation.

There are nine complete controls with identical row degrees and all true-key
edges retained. V0 retains K0 only, V3 retains all keys, V1 retains K1 plus one
additional key A in {0,2,3}, and V2 retains all keys except B in {0,1,3}.
Name these `guard_aA_bB`. Each removes exactly (3,2,1,0) key edges from
V0,V1,V2,V3 respectively, six total per head. The correct guard is `guard_a0_b3`;
the other eight are the complete set of matched alternatives. Every mask is
retained, with no random choice of a favorable control. These are matched on
edge counts, not on removed attention mass or activation-change norm. V0's
mask is necessarily identical across this control family.

The other six conditions are:

* `native`: original alternating layout and ordinary position IDs.
* `grouped_canonical`: unguarded grouped layout with canonical IDs.
* `grouped_noop`: reapply its unchanged L1 attention rows.
* `restore_keys`: transplant native L1 key residual states into the grouped
  model at L2 input, retaining unguarded value states.
* `guard_restore_keys`: correct guard plus native key-state restoration.
* `restore_values`: directly transplant native L1 value states at L2 input,
  leaving grouped key states unchanged.

This gives 15 conditions per seed, 90 model-condition cells. Native-state
transplants are explicitly oracle diagnostic controls, not deployable repairs
or independent discoveries. The correct guard itself uses no native activations,
targets or model fitting; it uses only the proposed key-visibility rule.

## What is guaranteed, and what is empirical

At L1, q/k/v are functions of token and position embeddings before any cross-token
computation. For canonical grouping, removing the newly exposed key edges leaves
each value exactly its original set of predecessors. Its attention result, MLP
and residual therefore equal the corresponding native L1 value state, up to
floating-point order. The final query sees the same complete input set in both
orders, so its L1 output is already invariant.

This identity is an algebraic implementation check. It is **not** the scientific
success criterion. Grouped L1 key states remain different because they do not
see earlier values that were visible during training. Their L2 keys/values can
still compete with restored value nodes. Thus final answer recovery by the guard
alone is not guaranteed. Restoring those key states as well makes the entire L2
input a permutation of native states; final-query restoration in that full oracle
is guaranteed and must be treated as another positive implementation check.

## Fresh data and prospective record

Generate 2,048 fresh association dictionaries with seed 8100001, balanced at
512 queries per pair index. Exclude every candidate whose 96 pair-order/query
variants overlap the audited earlier training, validation, E1, E2 or E3 inputs.
Exclude duplicate unordered dictionaries. Freeze actual input bytes and every
rejection before evaluation, without using outcomes.

Freeze this protocol, numerical forecasts, checkpoints, code and independent
preflight audit in a public commit. Verify remote manifest and exact Git blob
identities before the confirmation start or any trained forward. The controller
rechecks hashes and the exact remote commit, refuses an existing confirmation
directory and preserves partial runtime failures. No fitted or adaptive repair
is allowed within this test.

## Endpoints and gates

Save full output probabilities, predicted classes, correct-answer probabilities
and accuracy, per-case L1 and L2-input state differences, and all-head attention
diagnostics. Report all 15 conditions including oracle controls and every one of
the eight matched masks. The full claim requires all six seeds to pass:

1. **Validity.** Native accuracy ≥.95; grouped no-op target-probability error
   ≤1e-6; finite complete outputs. Within 1e-5: correct-guard L1 value and query
   states match native, correct-guard L1 key states match unguarded grouping,
   guard and direct-value restoration output probabilities match, and full
   guard-plus-key restoration output probabilities match native. Identity
   checks use every case (and every class for output distributions).
2. **Recovery.** Correct-guard accuracy ≥.95 within each of all four query
   strata. The paired 95% lower bound on `P(correct, guard)-P(correct, grouped)`
   must exceed .30 in every seed.
3. **Specificity.** The paired 95% lower bound on correct-guard probability
   minus the equally weighted mean probability under the eight other matched
   masks must exceed .05 in every seed.

The last gate concerns the mean of the complete control family. Report the
best individual control and every control effect; do not describe a passed
mean comparison as beating every possible mask or identifying a unique one.

Use 2,000 query-stratified multinomial dictionary-row bootstraps, seed 8200001,
sharing weights across all conditions and all seeds. Average the eight controls
within each dictionary before resampling. Individual Wilson 95% accuracy
intervals are not simultaneous intervals. Uncertainty is conditional on these
six selected checkpoints and input distribution, not a population of models.

Report per-query outcomes and the keys-only / guard / combined factorial
descriptively; no post-hoc new success gate. Report all 180 frozen numerical
forecasts (15 conditions × six seeds × probability and accuracy) and every
miss against the unchanged .15 tolerance. Numeric adequacy and causal gates
are separate predeclared questions. The no-op and oracle identities do not
inflate claims about predicting novel intervention effects.

## Interpretation

Selective final-output recovery would show that restoring training key
visibility is sufficient to repair this particular layout-induced failure,
with benefit beyond the mean of carefully matched sparsification controls.
It would not establish necessity of every removed edge, a unique complete
algorithm, spontaneous layout generalization, or a universal transformer
mechanism. Failure despite restored L1 value/query states would implicate
unrestored downstream context and refute this sufficiency claim.

Prior work already demonstrates positional binding and targeted steering; see
[E3's primary-source audit](../outputs/experiment3/RELATED_WORK.md). The narrower
new test is the causal-visibility failure boundary and its matched repair.
It does not establish field-wide novelty. E3's failed predictions remain public
regardless of what happens here. Any later refinement needs new prospective
data and registration; do not continue resampling the same claim until it passes.
