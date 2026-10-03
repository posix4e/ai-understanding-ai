# Experiment 4 prospective skeptical design review

This review precedes all trained-model E4 forwards. E4 is a new causal test motivated by E3's failure. E3's full claim failed in every seed, with 937/1,200 numeric forecast misses; its protocol, outcomes and interpretation must remain public and unchanged. Recovery under an additional intervention would not retroactively validate position-label sufficiency.

## Hypothesis and what is algebraically guaranteed

In canonical grouped order, every value receives its native token-plus-position input, but its first-layer attention can see keys that followed it in the training layout. Delete only edges from key Kj to value Vi when j>i: three edges at V0, two at V1, one at V2 and none at V3, in every head. Every value then has exactly its native predecessor-token set, including itself. Layer normalization and the MLP act independently at each token, and attention is invariant to reordering its visible key/value pairs.

Consequently, exact restoration of the first-layer value outputs is a mathematical manipulation check. The first-layer final-query output already has the same complete predecessor set and should also agree with native. These equalities are not discoveries established by success on the six checkpoints.

Final answer recovery is not algebraically guaranteed by this guard: the grouped first-layer key nodes still lack some preceding native value tokens. Their altered second-layer K/V can compete with restored value nodes. This makes final answer recovery the useful empirical endpoint. If final recovery fails despite valid algebraic checks, the six-edge explanation is insufficient; do not weaken the success criterion after observing that outcome.

## Complete edge-count control grid

Use all nine masks with the same retained-key count at each value and always preserve that value's own key:

* V0 retains K0 only; this row is identical in all nine masks.
* V1 retains K1 and one extra key A from {0,2,3}.
* V2 retains all keys except B from {0,1,3}.
* V3 retains all four keys.

The Cartesian product gives nine masks; `guard_a0_b3` is the causal-visibility guard and the other eight are shams. Every mask deletes six edges with the same per-row degree and applies to all heads. Keep value-to-value edges, key rows, query row, physical causal masking and trained parameters fixed. This grid controls sparsity and preservation of each correct-key edge. It does not match removed attention mass or vector perturbation norms, and V0 cannot provide specificity because its mask is forced.

The fixed grid has 15 conditions: these nine masks, native, canonical grouped, grouped no-op, native first-layer key restoration, guard plus native first-layer key restoration, and native first-layer value restoration. Report every condition. Compare the guard against the prespecified mean of eight shams; report the best sham descriptively without relabeling it as an independent selected competitor.

## Frozen decision criteria

The proposed primary conjunction requires all six existing seeds to satisfy:

1. Correct-guard original-answer accuracy >=.95 in each of the four query strata: 24 point-threshold cells overall.
2. Paired 95% bootstrap lower bound for guard-minus-canonical-grouped original-target probability strictly >.30.
3. Paired 95% bootstrap lower bound for guard-minus-mean-eight-shams original-target probability strictly >.05.

Native original accuracy >=.95 and the declared algebraic/no-op checks with maximum discrepancy <=1e-5 are validity conditions. Freeze the exact tensor or output compared for each algebraic check; report observed maximum discrepancies. A native failure or failed identity check invalidates the corresponding causal interpretation rather than permitting exclusion of a seed.

Use one balanced fresh sample of 2,048 dictionaries, 512 queries per pair index, for all conditions and seeds. Resample complete dictionary rows within query strata, sharing weights across every condition and model. Average the eight sham probabilities within each row before forming its guard contrast. Freeze bootstrap seed/count and use a strict lower-bound comparison. Report individual Wilson intervals for the 24 accuracy cells; these are not simultaneous confidence bands. The preregistered point-threshold conjunction is distinct from a multiplicity-adjusted population claim.

No new per-model fit or E4 discovery intervention should inform these thresholds. New numerical forecasts, if supplied, need a separate full error report; passing a causal gate must not hide failed confidence forecasts. Run the entire grid once after remote verification even if early results appear favorable or unfavorable.

## Oracle conditions and implementation audit

Restore activations at one clearly declared post-first-layer boundary, with original-to-grouped token alignment verified independently. `restore_values` should match the correct guard's value outputs and final-query result, because all other grouped nodes are unchanged. `guard_restore_keys` should restore every first-layer node to its native counterpart and therefore restore native final-query logits by permutation invariance. These conditions validate the manipulation and bound what can be restored using native activations; they are privileged oracles, not zero-information prediction baselines. `restore_keys` alone is a diagnostic of altered key-node competition, not a proof of necessity.

The implemented runner uses the existing attention-pattern hook but stably recomputes masked softmax from the current grouped Q/K for affected value rows; it does not divide underflowed retained probabilities. Verify with an independent random-weight implementation that masks the actual live model logits with negative infinity before softmax. Check exact deleted edges, row sums, unchanged unselected rows/heads, causal masking, position IDs, token alignment, downstream recomputation and oracle equalities. Tests that compare two copies of the same patch construction are insufficient.

The intervention acts on all heads by design. Report all-head diagnostics without selecting successful heads after outcomes. Attention correlations may guide interpretation but do not independently establish the reason for output recovery.

## Data, integrity and interpretation

Generate outcome-independent data from fixed seed 8100001 with exact query balance, four distinct keys/values and unique unordered association dictionaries. Exclude all 96 pair-order/query variants against the full historical input set, including E3 inputs, and freeze exact selected bytes before trained forwards. Independently audit target consistency, exclusion, code hashes and checkpoint identities. Preserve all E1–E3 frozen files. Remote readback of the E4 registration must precede any trained E4 condition, and execution must refuse overwriting an existing confirmation.

If the full conjunction passes, a justified narrow claim is that removing the specified newly exposed first-layer key edges is sufficient to restore retrieval in this grouped canonical layout, beyond the declared average sparsification control, across these six models and cases. It would identify a causal failure boundary for the insufficient E3 position-label account. It would not prove that those edges are the unique cause, that physical adjacency is irrelevant, that every mask of the same kind transfers, or that natural layout generalization has been learned. Mean-sham specificity does not imply superiority to every sham.

If recovery succeeds but specificity fails, generic sparsification remains a viable account under this design. If specificity succeeds but recovery fails, the mechanism has only partial explanatory scope. If only the full native-activation oracle succeeds, that is expected from its construction and does not support the guard hypothesis. Each outcome should be reported in the original terms, with any follow-up separately registered. Attention masking and positional steering are established intervention ideas; this study needs a bounded empirical contribution rather than a priority claim.
