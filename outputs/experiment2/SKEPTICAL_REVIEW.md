# Independent skeptical review: Experiment 2

**The frozen Experiment 2 predictor passes every registered validity, practical-accuracy and baseline-superiority gate in all six models.** The improvement is substantially broader than Experiment 1's single discriminating composition. Nevertheless, 50 of 552 novel condition-mean forecasts miss their individual 0.10 tolerance, and the successful predictor is a discovery-fitted second-order statistical model of intervention responses, not a uniquely recovered semantic circuit.

## Integrity and independent recomputation

Checked all **62 frozen file hashes** against the recorded Experiment 2 lock. They match. The lock names commit `f4b3f1d73952e4fa449b2d4bf1f139769b0d75ad`, recorded as remotely verified at **14:00:33 UTC** on October 3, 2026; the confirmation start record names that same commit at **14:00:49 UTC**. This review verifies consistency with the recorded remote readback, not an additional network readback.

Each released raw file contains all 145 conditions × three metrics, with 2,048 finite case values per array. Independently checked probability support, recomputed novel-condition means, family-balanced losses, errors, clean accuracy and no-op discrepancies directly from those arrays. I did not call the scorer or model runner.

For the superiority interval, independently used the algebraic identity that AI loss minus the minimum baseline loss equals the **maximum of seven paired loss differences**. Each difference is linear in the observed condition means after the common squared-mean term cancels. Applying the frozen shared multinomial bootstrap to those case-level differences reproduces the registered intervals within 1e-12. This independently checks pairing, family weights and best-baseline selection rather than merely reading the reported gate flags.

No new model interventions were run. No frozen source, forecast, input or result file was changed.

## Registered gates

| Seed | Cohort | Balanced RMSE | Novel means within 0.10 | Recomputed 95% AI-minus-best-baseline MSE interval |
|---|---|---:|---:|---|
| 0 | Existing | 0.05247 | 84/92 | [-0.01610, -0.01464] |
| 1 | Existing | 0.06822 | 83/92 | [-0.00763, -0.00601] |
| 2 | Existing | 0.07464 | 78/92 | [-0.01054, -0.00917] |
| 3 | New | 0.05565 | 85/92 | [-0.01307, -0.01190] |
| 4 | New | 0.04224 | 88/92 | [-0.01609, -0.01467] |
| 5 | New | 0.05934 | 84/92 | [-0.01409, -0.01269] |

All clean accuracies are 100%; maximum per-case no-op probability discrepancies are exactly zero. Every model has RMSE <=0.10, at least 74/92 novel forecasts within 0.10, and a superiority interval entirely below zero. The all-six conjunction therefore passes. Overall, **502/552 (90.9%)** novel means meet the individual tolerance. Familiar conditions and implementation controls do not enter the primary loss.

## Failures remain scientifically meaningful

The 50 misses consist of **19 pair** and **31 triple/rescue** forecasts; none is a cross-layer miss. Triple/rescue family RMSE is **0.10239 for seed 1** and **0.10407 for seed 2**. These exceed 0.10 descriptively, although family-specific RMSE was not an additional registered gate. Their favorable combined RMSE must not be summarized as uniformly accurate prediction of every family.

The largest error in each model illustrates failures in both directions:

| Seed | Condition | Forecast | Observed target probability | Absolute error |
|---|---|---:|---:|---:|
| 0 | L1 mean, corrupt heads 0+3 | 0.20252 | 0.44916 | 0.24664 |
| 1 | L1 zero, rescue head 0 | 0.94454 | 0.72798 | 0.21656 |
| 2 | L1 zero, corrupt heads 0+1 | 0.29890 | 0.55151 | 0.25262 |
| 3 | L2 mean, rescue head 0 | 0.42328 | 0.62261 | 0.19933 |
| 4 | L2 mean, rescue head 3 | 0.42886 | 0.55437 | 0.12551 |
| 5 | L1 mean, corrupt heads 1+2 | 0.26138 | 0.51402 | 0.25264 |

These are errors in condition means, not isolated unusual examples. Several are more than twice the accepted 0.10 tolerance and much larger than the observed mean's sampling uncertainty. They show that allocating all-head interaction residuals using singleton strengths is not an exact account of pair/triple behavior. The success claim is the registered aggregate/practical one, which explicitly permitted some misses.

## Baseline comparisons and robustness

The AI has lower point MSE than every tested baseline separately in the pair and triple/rescue families for every seed. Its **cross-layer forecasts are mathematically identical to case-logit-additive**, as identified before confirmation. That family cannot establish incremental value over that baseline. Moreover, count-logit is more accurate on cross-layer cases for seeds 4 and 5: AI RMSE is respectively 0.02474 and 0.01572 versus 0.01142 and 0.01196. Thus the predictor is not uniformly the best method in every family.

Two **post-hoc diagnostic checks**, not replacements for the registered test, probe dependence on a favorable subset:

* Remove each novel condition in turn, recompute equal-family loss over the remaining conditions, and reselect the best point baseline. The AI retains a lower point loss for all 92 removals in every seed.
* Remove each entire family in turn and weight the remaining two equally. The AI again retains a lower point loss in every seed, including when the cross-layer family is removed.

No additional confidence intervals or formal success gates were assigned to these diagnostics. They establish that the observed point advantage is not carried by a single condition or one indispensable family, unlike Experiment 1. The two per-case-logit baselines share the predictor's raw discovery information, so its advantage over the seven comparators cannot be explained solely by having access to paired case-level probabilities.

## What the result explains—and what it does not

The predictor is an **output-response surrogate**: it fits singleton and all-head probability changes on the log-odds scale, allocates residual interaction across head pairs, and extrapolates. Its causal input variables are explicitly controlled head interventions, which makes the prediction problem informative. However, it neither reconstructs the transformer's internal computation nor identifies unique meanings for its vectors. A successful statistical interaction rule can predict responses without establishing that its coefficients correspond to actual neural mechanisms.

Zero/mean corruption can create atypical states; resampling disrupts joint compatibility while preserving some task information. L1 active resampling preserves the query key and its position, so its weak effect does not prove that L1 head outputs are dispensable. Clean-head rescue demonstrates restoration under a specified corrupted background, not unconditional sufficiency. Head-output corruption retains projection bias, residual bypasses and MLP computation. Operational causal contribution is therefore better supported than strict natural necessity or a complete circuit account.

The six models share one small architecture, task family and selected training setting. The new seeds improve replication, but **each new model also received its own discovery-based fit**; this is not zero-shot transfer of a predictor fitted only to earlier models. Shared case bootstraps quantify uncertainty conditional on these checkpoints, calibration inputs, discovery fits and intervention grid. They do not estimate broad training-population or task-generalization uncertainty. The model form was selected after discovery, and seven baselines do not exhaust all plausible response models. Procedural shared-filesystem blinding remains weaker than enforced access isolation.

The supported advance is therefore specific: a frozen, discovery-guided numerical explanation predicts a broad reserved head-intervention grid accurately enough to meet declared tolerances and beats seven frozen alternatives in six trained models, including three new seeds. The failures identify where that approximation breaks. Neither passing the gates nor beating those baselines establishes unique mechanistic semantics or general AI self-understanding.
