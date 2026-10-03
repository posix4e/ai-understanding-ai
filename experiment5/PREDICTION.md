# Experiment 5 prospective predictions

Prepared 2026-10-03 from published E3/E4 results and the outcome-independent E5 condition specification. The predictor has not read E5 checkpoint metrics, evaluation inputs, or outcomes and has performed no E5 model forwards. Training seeds 6–11 are fixed in advance, with no replacements. These forecasts must be frozen and remotely verified with the protocol, code, checkpoints, and inputs before confirmation. Blinding is procedural on a shared filesystem, not enforced by operating-system isolation.

## Explanation and its limits

E4 supports a partial repair: restoring native first-layer value states usually recovered final answers despite altered key states. This does not imply that key states are irrelevant. E4 includes one guard error removed by restoring keys, several highly effective alternative masks, and failures of the predictor's local-row mixture model. Those counterexamples remain part of the explanation.

E5 tests whether the same partial repair transfers prospectively to six new training seeds and a complete class of 105 layouts. Canonical position IDs stay with their tokens; each key precedes its own value, value order is preserved, and the final query remains at physical position 8. The correct guard removes visible keys Kj with j>i from each first-layer value row Vi. Each value and the final query then have their native first-layer states, up to floating-point error. Changed key states remain available to layer 2. The resulting final-answer accuracy is an empirical prediction, not a restoration theorem.

All 729 feasible masks across these layouts are tested: 105 correct masks and 624 shams. A sham retains the true key and the same number of key predecessors in every value row as the correct guard. These controls match row degree, not removed attention mass or activation norm. Three layouts have no alternative mask. Specificity averages the shams within each of the 102 eligible layouts, then averages layouts equally. It does not average all 624 shams as if layouts with more alternatives deserved greater weight. A mean-control advantage does not establish uniqueness or superiority over every sham.

The 2,415 remaining own-key-before-value layouts violate value order. Four conditions are evaluated on the first 256 balanced cases: unguarded, keyguard, logical_prefix, and own_key_self. The logical-prefix mask removes physically visible tokens whose logical coordinates exceed the value's native prefix; it cannot supply absent predecessors. Across this complete boundary family, earlier keys as well as earlier values can be absent. Success therefore supports tolerance to missing native context generally, not specifically dispensability of earlier value tokens. The own-key-and-self mask makes first-layer value states invariant to these layouts; it does not guarantee accurate answers or invariance of the remaining key states.

The secondary edge panel adds each forbidden edge Vi←Kj, i<j, individually to the correctly guarded grouped layout. The proposed mechanism is off-target competition: the added key can make Vi compete for query Kj, although the correct value Vj remains restored. This predicts a named wrong answer, rather than merely degraded accuracy. An explanation based only on the queried value's own state would not predict this effect. Failure of this panel leaves the primary transfer decision unchanged and limits the stronger competition explanation.

## Numerical forecasts

The JSON freezes twelve aggregate point forecasts for each of seeds 6–11, for 72 numbers in total. They are identical across seeds because no E5 model-specific evidence is available. They are judgments informed by E4 and the intervention structure, not fitted probabilities or calibrated prediction intervals. The boundary and edge forecasts are substantially less supported than the primary repair forecast.

| Endpoint | Point forecast |
| --- | ---: |
| Native mean original-answer probability | .999 |
| Native accuracy | 1.000 |
| Correct-guard mean original-answer probability over 105 layouts | .995 |
| Correct-guard accuracy over 105 layouts | .998 |
| Minimum correct-guard accuracy across 105×4 layout/query cells | .980 |
| Native minus correct-guard mean probability over 105 layouts | .004 |
| Correct guard minus mean complete sham family, averaged over 102 eligible layouts | .080 |
| Boundary logical-prefix overall accuracy | .985 |
| Boundary own-key-and-self overall accuracy | .980 |
| Fraction of boundary layout/query cells with logical-prefix accuracy ≥.95 | .980 |
| Fraction of boundary layout/query cells with own-key-and-self accuracy ≥.95 | .970 |
| Edge-panel named-competitor difference of changes | .120 |

Every point forecast has the same practical absolute-error tolerance .05. Report all 72 observed values, errors, misses, MAE and RMSE, with per-seed results. This tolerance is not a coverage probability and does not replace the scientific gates below. There are no frozen numerical forecasts for every individual mask, boundary baseline, or edge; their complete outcomes are still reported. E4 does not identify those individual responses, and inventing a detailed table would give unsupported precision.

The original answer is the value paired with the queried key in the underlying dictionary. Probability always means its softmax probability, unless the endpoint explicitly names an alternative displayed value. Accuracies use the full 16-class argmax. Chance over all classes is 1/16; an exchangeable choice among the four displayed values is 1/4. Neither is a mechanistic matched control.

## Frozen decisions

For each new seed, primary recovery requires accuracy ≥.95 in **every** one of the 105×4 layout/query cells, and the paired 95% bootstrap upper bound for native minus correct-guard mean original-answer probability, averaged equally across 105 layouts, must be ≤.01. Primary specificity requires the paired 95% lower bound for correct guard minus the within-layout mean of all shams, averaged equally over the 102 eligible layouts, to be strictly >.02. The full primary claim requires validity, recovery, and specificity in all six fixed seeds. A failed seed or cell is retained; pooled success cannot replace this conjunction.

Secondary boundary success is assessed separately for logical_prefix and own_key_self: overall accuracy, equally weighted over 2,415 layouts and balanced queries, must be ≥.95 in each seed, and an all-six claim requires all six seeds to pass. Report the fraction of all 2,415×4 layout/query cells meeting .95, including every failure; no additional fraction threshold is a scientific gate. These secondary decisions cannot rescue a failed primary claim.

For edge (i,j), restrict the directional contrast to cases querying Kj. Let Δ mean the added-edge condition minus the correctly guarded grouped condition on the same case. The case contrast is ΔP(Vi) minus the mean ΔP of the other two displayed wrong values, excluding Vi and the correct Vj. Average cases within the query stratum, then average the six edge means equally. Secondary edge success requires its paired 95% lower bound to be strictly >.05 in each seed; an all-six claim requires all six. Retain all query strata and individual edges descriptively, including q=0, which is ineligible for this directional endpoint. There is no requirement that each edge separately pass.

## Validity and uncertainty

Validity requires native accuracy ≥.95; finite, complete, internally consistent outputs for all masks; maximum full-16-class no-op probability error ≤1e-6; and the following maximum absolute errors ≤1e-5: correct-guard first-layer value states versus aligned native states on all primary cases; first-layer query states versus native and unedited first-layer key states versus their unguarded-layout counterparts for every mask in every panel; full-oracle 16-class probabilities versus native; boundary own-key-and-self first-layer value states versus the native own-key-and-self reference. Oracle and state identities are manipulation checks, not empirical discoveries. A failed validity gate blocks the relevant primary claim rather than authorizing seed replacement.

Use 2,000 paired dictionary-row multinomial bootstrap replicates, stratified within the four query indices. Primary and edge analyses use N=1,024 and random seed 10200001; the first-256-case boundary panel uses a separate generator with seed 10200002. Within each panel, reuse the same row weights across all layouts, conditions, and models. Aggregate layouts and shams within each row before computing primary contrasts. For each edge, use the same global draw's normalized weights in its eligible query stratum, then average six edge means equally; repeated use of a query stratum does not create independent samples. No model bootstrap or population-level generalization to all training seeds is claimed. Pointwise intervals are not simultaneous bands.

All new seeds and all layouts remain visible. The study tests externally supplied causal masks in this model/task family. It does not establish spontaneous layout understanding, a unique algorithm, individual-edge necessity, universal transformer behavior, or field-wide novelty. Results that motivate a different mechanism require a separately registered study.
