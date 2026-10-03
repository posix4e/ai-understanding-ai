# Experiment 4: prospective prediction of a partial attention-context repair

**E3's completed outcomes are discovery evidence for this separately registered E4.** E3 failed its six-model mechanistic claim and its numeric forecasts were inaccurate; those results remain unchanged. I used E3's published results and saved summary to calibrate unchanged references only. I have not inspected E4 inputs or outcomes, run a new model evaluation, or observed any guard condition. There is no E4 discovery phase. The new forecasts below are untested theoretical extrapolations, not repairs to E3's old predictions.

## Hypothesis and nontrivial prediction

Grouped canonical tokens `K0 K1 K2 K3 V0 V1 V2 V3 Q`, with IDs `[0,2,4,6,1,3,5,7,8]`, expose earlier-logical values to later-logical keys that were causally absent in the interleaved training layout. The proposed cause of much of E3's failure is disruption of the L1 value representations by these extra competitors. At L1, mask attention from value query Vi to key source Kj for j>i, across every head, and renormalize the retained row. Preserve all other edges and the true key Ki.

This **correct prefix guard** makes every L1 value's allowed set of token-plus-position inputs equal to its original interleaved set, up to ordering. Therefore L1 value outputs equal their native counterparts algebraically; the query's L1 output is already invariant because it sees the complete input set. Softmax source-order symmetry plus positionwise residual/MLP computations establish these identities for arbitrary weights. Their verification is an implementation/manipulation check, not a learned discovery.

The unguaranteed claim is that this partial repair restores behavior. L1 key-token states remain changed: grouped keys cannot see preceding values that existed before them in the original layout. They remain competitors for the final L2 query. Thus final predictions need not recover even when all guarded L1 value states match perfectly. E4 tests the stronger, falsifiable proposal that value-site repair is sufficient in these trained models despite those remaining differences.

## The complete 15-condition design

The nine guard IDs are `guard_aA_bB`, with A in `{0,2,3}` and B in `{0,1,3}`. They retain the following **key sources** at L1 value-query rows; all value sources and other previously permitted sources remain as in grouped canonical:

| Value-query row | Retained key sources |
|---|---|
| V0 | K0 |
| V1 | K1 and KA |
| V2 | all keys except KB |
| V3 | all four keys |

Every guard retains the true key and deletes `(3,2,1,0)` key edges across rows, six total. The correct guard is **`guard_a0_b3`**. The other eight are all degree-matched shams; none is chosen based on outcomes. They match edge counts per row, not removed attention mass or activation norms.

The six additional conditions are `native`, `grouped_canonical`, `grouped_noop`, `restore_keys`, `guard_restore_keys`, and `restore_values`. `restore_keys` patches native L1 key outputs into the grouped L2 input; `restore_values` patches native L1 value outputs into the otherwise unguarded grouped L2 input; `guard_restore_keys` combines the correct guard and the native key-state patch. All patches align token identities across physical layouts and leave unselected components unchanged.

`restore_values` must produce the same full 16-class final probability vector as the correct guard. `guard_restore_keys` is a **full-restoration oracle**: it makes all L1 states native after alignment, so the final query's L2 calculation is equivalent to native. Neither oracle equivalence is an independent scientific success. In contrast, the effect of `restore_keys` alone is not guaranteed.

## Frozen numerical forecasts

`predictions.json` contains `expected_probability` and `expected_accuracy`, each keyed by seed 0–5 and the 15 exact condition IDs: **180 numbers**. The endpoint is mean original-target probability or original-answer accuracy on the new E4 cases. E3 reference means are recorded with their source hash; no guard output was used to fit anything.

For every seed:

| Condition | Mean target probability forecast | Accuracy forecast |
|---|---:|---:|
| native | E3 native probability | E3 native accuracy |
| guard_restore_keys | same E3 native probability | same E3 native accuracy |
| grouped_canonical, grouped_noop | E3 grouped-canonical probability Gp | E3 grouped-canonical accuracy Ga |
| restore_keys | Gp | Ga |
| correct guard, restore_values | **0.98** | **0.99** |
| Any guard aA_bB | `r*0.98+(1-r)*Gp` | `r*0.99+(1-r)*Ga` |

Here `r=(2+I[A=0]+I[B=3])/4` is the fraction of L1 value rows whose exact native predecessor set is restored. V0 and V3 are restored under every mask; V1 is restored exactly when A=0, and V2 when B=3. Thus the correct guard has r=1, four shams have r=0.75, and four have r=0.50. Queries must be balanced across four pair indices, as fixed in the input protocol.

The mixture extrapolates that a query with its value row repaired mostly behaves like the high-recovery endpoint, while an unrepaired row behaves like the E3 grouped reference. This is an **untested local-query approximation**, not an implication of the state-equality theorem: corrupted value rows can attract other queries, and a wrong mask may help despite failing exact equality. The conservative `restore_keys` forecast of no average improvement commits to the hypothesis that key-state repair alone does not substantially fix the damaged value pathway. It too may fail.

The 0.98/0.99 guard confidence/accuracy forecasts are deliberate strong commitments, not estimates inferred from additional runs. Use **absolute error tolerance 0.15 for every numeric forecast**, with bounds intersected with [0,1]. Report every error, count of misses and RMSE. These are practical bounds, not calibrated intervals. Numeric forecasting and the mechanistic conjunction below are separate assessments; neither may hide failures in the other.

## Scientific gates and why they remain demanding

The primary mechanistic claim requires **all six fixed models** to pass manipulation validity, rescue and specificity:

1. **Manipulation validity:** correct-guard L1 value outputs and query output match aligned native outputs with maximum absolute error <=1e-5; correct-guard L1 key outputs match **unguarded grouped-canonical** key outputs within 1e-5; correct guard and `restore_values` final full 16-class probability arrays differ by at most 1e-5; full oracle and native final full 16-class probability arrays differ by at most 1e-5. Probability-array identities use the maximum absolute difference over every case and all output classes, not only the target. Require native accuracy >=95%, finite outputs, and grouped no-op maximum original-target-probability error <=1e-6. These are theorem/implementation checks, not discoveries.
2. **Behavioral rescue:** correct-guard original-answer accuracy is >=95% in **every query-index stratum of every model**—24 model/stratum cells. For each model, the paired 95% lower confidence bound on mean original-target probability improvement over `grouped_canonical` exceeds **0.30**.
3. **Average-control specificity:** for each model, the paired 95% lower bound on correct-guard original-target probability minus the **equally weighted mean of all eight shams** exceeds **0.05**. Compute the eight-sham mean within each case before paired resampling.

The specificity gate does **not** require beating the best sham. Always report the best sham, all eight individual contrasts, their per-query outcomes and the full distribution. A successful average contrast supports benefit relative to the complete matched control family, not uniqueness of the training-prefix mask or superiority to every valid sparse circuit. A competing mask may reveal an alternative sufficient repair. This interpretation is part of the prospective claim, not an excuse introduced after a result.

The >=95% requirement in every query stratum avoids hiding one broken association behind a high average. The probability-gain requirement remains substantial even for E3's strongest grouped-canonical model. These gates need not be weakened merely because other shams can succeed: the intended claim is strong partial repair plus an average identity-specific benefit, with every counterexample retained.

Use fresh identity-audited inputs and **2,000 query-stratified multinomial bootstrap replicates with RNG seed 8200001**, as fixed in E4's protocol. Resample whole dictionary rows within balanced query strata, retaining all conditions together and identical resampling weights across models. Condition all uncertainty on the six checkpoints; report seeds individually and descriptive ranges. No seed, case, mask or query position is removed for a scientific failure. All bootstrap and numerical validity details must be frozen before evaluation; they are aligned here with the final E4 protocol.

## Falsification and scope

Failure of a structural identity signals an implementation or modeling-assumption problem. Failure of behavioral rescue **despite** valid identities rejects the claim that correcting these L1 value contexts sufficiently explains the grouped-layout deficit. Rescue without specificity supports a useful repair but not the claim that this particular prefix structure matters relative to equally sparse controls. A large effect of key-only restoration or a superior sham limits any exclusive account. High aggregate performance with one failed query stratum does not satisfy the registered conjunction.

A positive result would support a **training-context boundary for a partial causal pathway**: in these models, restoring a specific subset of predecessor relationships can recover performance despite other intermediate states remaining different. It would not show spontaneous grouped-layout competence, discovery of a new binding algorithm, or that the six deleted edges individually cause every failure. The intervention explicitly supplies known training structure. The architectural restoration identity itself is not empirical novelty. E3 remains a failed prediction; E4 can explain a failure boundary only through this separately frozen and independently evaluated test.
