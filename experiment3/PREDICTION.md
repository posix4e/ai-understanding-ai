# Experiment 3: prospective prediction of coordinate-induced answer rebinding

**There is no E3 discovery phase. No E3 input inspection or model evaluation preceded this prediction.** The predictor uses the original model architecture and previously authorized E1/E2 discovery evidence, without E2 confirmation outcomes. It has not read E3 input arrays or executed model forwards. All six model seeds 0–5 receive the same numerical forecasts, with no new per-model fit. The supplied position-coordinate hypothesis motivates this experiment; a favorable result would not be independent invention of that hypothesis.

## Mechanistic commitment

The hypothesis is that learned position codes help bind logical key slot `2j` to value slot `2j+1`. Changing the position code while keeping token content fixed can therefore change which answer a query selects. The competing strict physical-adjacency account predicts that rearranging these codes alone will not restore ordinary binding in a grouped layout or cause the specifically prescribed wrong answer.

This contrast tests **dependence on assigned position codes**. It cannot distinguish a learned table pairing coordinates `(0,1),(2,3),(4,5),(6,7)` from an operation recognizing the preceding coordinate, nor identify the responsible attention head. The causal mask remains based on physical token order; no logical-position mask or model-weight change is allowed.

## Conditions and exact target identity

Every case contains four distinct pairs `(K0,V0)…(K3,V3)` and query `Q=Kq`. Original native order is `K0 V0 K1 V1 K2 V2 K3 V3 Q`; grouped order is `K0 K1 K2 K3 V0 V1 V2 V3 Q`. Both have length nine and Q at physical position 8. Canonical grouped IDs are `[0,2,4,6,1,3,5,7,8]`; native grouped IDs are `[0,1,2,3,4,5,6,7,8]`.

For all 24 permutations `pi` of `[0,1,2,3]`, pi maps an original pair index to its assigned logical slot. The full fixed grid uses:

* **Coherent mapping:** Ki gets ID `2*pi[i]`, Vi gets `2*pi[i]+1`, Q gets 8. Predict the original answer **Vq**.
* **Value-only mapping:** Ki keeps ID `2*i`, Vi gets ID `2*pi[i]+1`, Q gets 8. Predict the value at index **inverse(pi)[q]**. This is an exact per-case answer rule.

For example, `pi=1032` swaps slots 0↔1 and 2↔3. Query K0 predicts V1 under value-only mapping and V0 under coherent mapping. Neither a generic wrong answer nor merely low original-answer probability satisfies the rule.

The identity maps in both families collapse into `grouped_canonical`. The remaining 23 permutations per family are named `coherent_XXXX` and `value_XXXX`, with digits in lexicographic permutation order. Add `original_native`, identical `original_noop`, `grouped_native` and `grouped_canonical`, for **50 total conditions**. All nine derangements in the value-only family and their nine coherent matches form the primary mechanistic comparison. The 14 nonidentity permutations having fixed points in each family are secondary tests; keep all, with no outcome-based selection.

The `slot_target` metric uses inverse(pi)[q] only for `value_XXXX`; it uses q for all other conditions. `grouped_native` has no defined slot-binding target in this hypothesis: its slot metric duplicates the original target **solely as a reference**, not as a theoretical claim.

## Numeric forecasts frozen before outcomes

`predictions.json` contains mean original-target probability, mean slot-target probability, slot-target accuracy and original-target accuracy for all 50 conditions in all six models (**1,200 numeric entries**). Its authoritative dictionaries are `expected_original_probability`, `expected_slot_probability`, `expected_slot_accuracy`, and `expected_original_accuracy`, each keyed by model seed and exact condition ID.

| Condition | P(original) | P(slot target) | Original accuracy | Slot accuracy |
|---|---:|---:|---:|---:|
| original_native, original_noop | 0.99 | 0.99 | 0.995 | 0.995 |
| grouped_native | 0.25 | 0.25 | 0.25 | 0.25 |
| grouped_canonical | 0.95 | 0.95 | 0.97 | 0.97 |
| Every coherent permutation | 0.95 | 0.95 | 0.97 | 0.97 |
| Value-only permutation with f fixed points | `0.03+0.92*f/4` | 0.95 | `0.01+0.96*f/4` | 0.97 |

The original-target mixture uses exactly balanced queries: 512 cases for each q among **2,048** cases. When pi fixes q, original and slot targets coincide; otherwise the forecast is original probability 0.03/accuracy 0.01 versus slot probability 0.95/accuracy 0.97. Thus all derangements forecast original probability **0.03**, one-fixed-point permutations **0.26**, and two-fixed-point permutations **0.49**. These cases are not individually independent replications of the learned model.

The forecasts are strong prospective estimates under the proposed mechanism, not values inferred from shifted observations. In particular, grouped/native 0.25 is an uncertain displayed-value heuristic; it is not mathematically forced by the original discovery evidence. Use practical error tolerance **±0.15 on every probability/accuracy forecast cell**, including grouped/native, intersected with [0,1]. These are error bounds, not calibrated predictive intervals. Report all numeric forecast errors, misses and RMSE with the original tolerance unchanged. **Missing an arbitrary confidence forecast does not automatically erase a selective answer-rebinding finding**; the mechanistic gates below are separate and must be reported separately.

## Main scientific gates

For each of all six models, first require native original-answer accuracy >=95%, finite output arrays, and maximum absolute original-target-probability discrepancy <=1e-6 between `original_native` and `original_noop`. Preserve every seed even if this validity check fails. A global claim requires passing the appropriate gates in every seed, with individual results and ranges visible.

For strong evidence of **specific named-answer rebinding under trained-parity-preserving position coordinates**, require:

1. In **every seed × each of nine value-only derangements**, slot-answer accuracy is **>=90%**; in **every seed × each matched coherent condition**, original-answer accuracy is **>=95%**.
2. In **each of six seeds**, the paired 95% lower confidence bound on the **equally weighted nine-derangement mean** of `P(slot)-P(original)` exceeds **0.80**. This adds a demanding confidence requirement beyond answer accuracy.

The fixed native-accuracy/no-op validity requirements above also apply. This is a conjunction over the entire primary grid; no favorable subset is sufficient. Main conclusion: the model follows the specified wrong-answer identity while the matched coherent mapping preserves the original answer. It is not a claim that any position perturbation harms or helps.

Report `P(original,grouped_canonical)-P(original,grouped_native)` and its paired confidence interval **separately and descriptively**. Failure of native grouping is not required for the primary rebinding finding. Report per-permutation coherent-minus-value original probabilities, competing displayed-value probabilities from the frozen full-class probability output, and canonical accuracy as additional descriptive evidence. There is no additional hidden gate based on these contrasts. The numeric forecast assessment, with every original forecast/error and RMSE retained, is separate from the mechanistic gates.

Use **2,000** paired bootstrap replicates, RNG seed **7200001**, resampling whole cases within each query-index stratum and preserving all 50 conditions together. Use the same resampling weights across models, since they share cases. The intervals condition on these checkpoints and the fixed input design; they are not uncertainty over the model-training population. Report individual condition results as well as the aggregate contrasts, and do not turn a favorable subset of permutations into the primary result. No simultaneous-coverage claim is made for all displayed intervals.

## Explicit answer-level baselines

Freeze the following algorithms as complete distributions over the 16 value classes before evaluation. They require no fitting. Their nominal target probabilities below specify the rules; the primary comparator report measures answer fidelity against the actual model argmax on the same complete case/condition set. For randomized rules use expected match probability rather than sampling answers or breaking ties arbitrarily. Do not choose a weak comparison afterward. These are answer-identity hypothesis controls, not fitted probability-prediction baselines.

* **Semantics-only original:** assign probability one to Vq in every layout and coordinate map. For value-only conditions its expected slot probability under balanced queries is f/4, and zero on all derangements.
* **Exchangeable displayed values:** assign 1/4 to each of the four displayed values and zero elsewhere. Its original and slot probabilities are always 0.25. If an accuracy comparator is needed, report expected randomized accuracy, not an arbitrary argmax tie-break.
* **Uniform vocabulary:** assign 1/16 to each value class, giving original/slot probability and expected randomized accuracy 0.0625.
* **Forward-permutation structural rival (secondary):** for value-only mapping predict V_pi(q), rather than V_inverse(pi)(q); in reference/coherent conditions predict Vq. Evaluate the directional distinction only on the prespecified **noninvolutive permutations and query cases for which pi(q) differs from inverse(pi)(q)**. Report both named-answer accuracies against the observed model argmax, the paired difference of their match indicators, and the corresponding paired probability difference from the frozen full-class probability output. These remain secondary descriptive comparisons, not additional primary gates. The complete noninvolutive grid consists of six 4-cycles and eight 3-cycles; within a 3-cycle exclude only its fixed query index from this directional comparison. These exclusions define the structural comparison before outcomes and do not remove cases from primary scoring or reporting. Involutions cannot distinguish the two rules and must not be presented as evidence of direction. This is an explicitly secondary test, not an extra primary gate.

The hypothesis's own exact algorithm—coherent→Vq, value-only→V_inverse(pi)[q]—is supplied in advance and should be evaluated directly by its slot-answer accuracy. High agreement does not show the network uses the same internal algorithm rather than an observationally equivalent operation.

## Falsification, limitations and integrity

Failure to rescue grouped canonical IDs refutes the strong transfer prediction but does not uniquely establish physical adjacency. A changed physical layout alters causal exposure and competition among earlier tokens despite unchanged sequence length and query index. If coherent mappings fail too, wrong-map damage cannot be called selective rebinding. If value-only permutations lower original accuracy without tracking inverse(pi), the specific binding explanation fails. Partial permutations additionally test whether fixed-point queries retain original answers while displaced queries follow the prescribed target; report these subsets without redefining the primary derangement grid.

Keep exact identity-audited inputs, all model checkpoints, condition specifications, forecast files, target algorithms, scoring code and bootstrap plan under a verified remote preregistration commit before any E3 forward. Audit or selection may depend only on input identities and declared balance/uniqueness rules, never model behavior. No seed, case, permutation or error is dropped. Failed or ambiguous outcomes are valid scientific results; extensions require a new prospective record. Blinding is procedural on a shared filesystem, not OS-enforced. No claim extends from this coordinate-permutation grid to every physical token order, to unique circuit semantics, or to general AI self-understanding.
