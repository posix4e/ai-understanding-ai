# Experiment 2: frozen prediction of unseen head combinations

The discovery evidence supports head-specific, partly redundant contributions to routing and answer content. It does **not** identify how those heads interact when two or three are corrupted. This predictor makes that missing interaction assumption explicit and permits a clear scientific failure. It forecasts original-target probability for every condition in every fixed model, not the answer on a particular unseen case.

## Evidence and interpretation

All six models answer the 1,024 discovery cases correctly when clean, with mean target probability approximately 0.9993–0.9994. L1 mean/zero corruption is highly head-specific: seed 0 is sensitive mainly to heads 0 and 3, seed 2 mainly to heads 1 and 0, and seed 4's head 2 alone reduces mean target probability to about 0.287 with mean replacement (all-head replacement gives 0.175). Head labels are local to a model, not comparable learned identities across seeds. Seed 3 is especially sensitive to head 0; seed 5 to head 1. Seed 1 has more graded contributions.

L1 resampling preserves performance even when all four heads are replaced, although mean/zero replacement of all four substantially impairs it. These donors preserve the queried key and its pair position. Thus resampling may preserve the task-relevant routing information while changing other context; successful resampling is not evidence that L1 outputs are dispensable. L2 singles give graded damage, all-head active resampling makes original-target probability approximately zero, and same-answer matched resampling remains nearly clean. This is consistent with distributed answer-content contributions at L2 and recipient residual bypasses. Whole-head-output corruption retains the projection bias, MLP and residual stream; it is not removal of an entire layer.

These observations refine the previously supplied routing/content hypotheses. They do not establish literal key semantics, unique circuitry or natural head necessity. Mean and zero corruptions may be atypical activations; resampled heads can be incompatible with the remaining recipient computation. The observed all-head endpoint also leaves the shape of the response between one and four corrupted heads undetermined.

## Fixed numerical rule

`predictor.py` is a deterministic numerical program using only the authorized discovery NPZ probability arrays, discovery summary and fixed condition specifications. It does not import Torch, load checkpoints or call an intervention runner. `predictions.json` contains **145 forecasts × six seeds = 870 numbers**, all in [0,1], plus fitted quantities, fixed constants and source hashes. No novel pair, sole-head rescue or cross-layer result was used. All forecasts for familiar conditions and controls simply retain their observed discovery means.

For each seed, layer and corruption type separately, let `p0(t)`, `ph(t)` and `p4(t)` be the original-target probabilities on discovery case `t` when clean, when head `h` alone is corrupted, and when all four heads are corrupted. Define `ell(p)=log(p/(1-p))` after clipping p to `[1e-6,1-1e-6]`. Set:

```
x0(t) = ell(p0(t))
dh(t) = ell(ph(t)) - x0(t)
strength[h] = max(0.001, -mean_t dh(t))
w[h] = strength[h] / sum_h strength[h]
R(t) = ell(p4(t)) - x0(t) - sum_h dh(t)
q(S) = sum_{h<j, h,j in S} w[h]w[j] / sum_{h<j} w[h]w[j]
xS(t) = x0(t) + sum_{h in S} dh(t) + q(S) R(t)
forecast(S) = mean_t sigmoid(xS(t))
```

The floor 0.001 makes pair weights defined even for nearly harmless corruptions and prevents deletion of a head on the basis of a noisy negative average loss. It is a fixed numerical regularizer, not a parameter chosen against reserved interventions. The sigmoid exponent is clipped to [-50,50] for numerical stability. No fitted hyperparameter search or alternative-predictor selection was performed.

This is a second-order interaction model on log odds. Singleton changes contribute directly. The known all-head departure from their additive sum is distributed across head pairs in proportion to their operational strengths. It matches the clipped clean, singleton and all-head endpoints case by case: `q` is zero for fewer than two selected heads and one for all four. There are no free pair-specific coefficients and no cubic/four-way term. The assumption is that stronger single-head disruptions receive more of the unobserved interaction residual. It is plausible for overlapping evidence pathways, but is **not identifiable from the available measurements**.

For sole-head rescue, `S` is the complement of the retained head: the intervention is exactly corruption of the other three heads. For cross-layer conditions, add the L1 and L2 singleton log-odds changes to `x0` and apply the sigmoid before averaging. No cross-layer interaction term is fitted because none has been observed. This forecast does not imply the evaluator clamps the unselected L2 heads to their clean values: those heads must recompute under the L1 intervention. Instead it makes a testable approximation about the resulting output evidence.

Examples illustrate the frozen distinctions. Seed 0 mean-corrupting L1 heads 0+3 is predicted near **0.203**, while corrupting heads 1+2 is predicted near **0.999**. Seed 4 mean-rescuing L1 head 2 alone is predicted near **0.999**, while rescuing any other L1 head is near **0.205**. For seed 0 L2 active resampling, two-head forecasts span about **0.454–0.671**, and sole-head rescue forecasts about **0.067–0.241**. These are prospective numeric extrapolations, not hidden observations. The full JSON, rather than rounded examples, is authoritative.

## Bounds and prespecified decisions

Each condition has a practical acceptance interval `[max(0,prediction-0.10), min(1,prediction+0.10)]`. These are ten-percentage-point error bounds on **mean target probability**, not predictive intervals with calibrated coverage, intervals for accuracy, or claims about every example. They intentionally leave room for extrapolation error beyond discovery sampling noise.

The primary novel score is MSE averaged equally across three groups—36 pairs, 24 sole-head rescues/three-head corruptions, and 32 cross-layer conditions—giving each group weight 1/3. Its square root is the primary RMSE. Familiar/reference/control cells do not enter this score. **Practical predictive success requires, in every one of six seeds, balanced RMSE <=0.10 and at least 80% of its 92 novel forecasts within 0.10 (at least 74).** Interpretation as a valid model test additionally requires clean accuracy >=95% and runtime no-op maximum probability error <=1e-6 in each model. These validity checks are distinct from the two descriptive forecast-adequacy gates. A global practical-success claim requires both adequacy gates and validity checks in all six models; partial successes must name the models/groups that pass and fail. Report seeds 0–2 and new training seeds 3–5 separately, without omitting any model.

Incremental predictive value is a separate, more stringent gate: the paired-bootstrap 95% interval for AI balanced MSE minus the **minimum loss among all seven frozen baselines** must have upper endpoint strictly below zero in every seed. The minimum is recomputed inside each bootstrap replicate, as specified by `score.py`; a convenient baseline is not selected for comparison afterward. Report the best point-estimate baseline, every baseline loss, the gap and its interval. Practical success can occur without superiority; superiority can occur while the absolute forecasts are still poor. Only passage of both categories supports the combined claim.

Bootstrap 2,000 resamples using NumPy multinomial case counts, seed **6200001**, retaining all interventions and recipient/donor relationships together. Use identical resampling weights across models because their cases are shared. These intervals condition on the checkpoints, calibration, discovery data and fitted forecast parameters; they do not include uncertainty from fitting or training. Six models are not thousands of independent training replications.

## Falsification and limits

The model can fail if a pair has unusually strong interference or compensation unrelated to singleton strength, if three-head effects require a higher-order term, if a small singleton effect conceals a jointly indispensable route, or if upstream corruption changes downstream head behavior enough to invalidate the cross-layer additive-evidence approximation. Systematic misses concentrated in the triple/rescue family would specifically challenge the second-order allocation; misses concentrated in cross-layer conditions would challenge the zero cross-layer-interaction assumption. Preserve these failures even if a different baseline succeeds. Accuracy and target-versus-best-other logit margin remain secondary checks; mean probability agreement alone does not imply identical decisions.

The seven baselines are zero-effect, additive probability changes, multiplicative probability retention, count-linear interpolation, count-logit interpolation, **case-logit-additive**, and **case-logit-count**, as fixed in `baselines.py`. Count interpolation retains the mean of the selected singleton effects; it is not perfectly identity-blind. The first five use discovery condition means. The last two were added **after discovery and before confirmation** specifically to address the AI predictor's access to paired per-case distributions. They use the same raw discovery probabilities: case-logit-additive sums per-case singleton log-odds shifts before applying sigmoid and averaging; case-logit-count interpolates on each case's log-odds scale between the selected singleton average and the all-head endpoint, then combines cross-layer changes additively relative to clean. Thus the strengthened comparison includes distribution-aware empirical rules as well as simpler summaries. They do not fit the predictor's head-weighted pair residual allocation. Any advantage is still a test of the complete forecasting rule, not proof that its mechanistic interpretation is uniquely correct. Log-odds nonadditivity likewise does not establish a neural interaction.

This model form and its regularizers were chosen after discovery and before confirmation. Existing E1 findings and supplied mechanism hypotheses informed interpretation; E2 is an extension, not independent invention of those hypotheses. No external paper is claimed to recommend this particular predictor. A favorable result would establish useful prospective forecasts on this fixed corruption/rescue grid; a negative result would show the selected explanation is quantitatively inadequate even if its broad routing/content interpretation remains compatible with discovery.

The predictor accessed only the declared implementation, design review, scoring definitions, training metadata and discovery evidence. It did not inspect frozen confirmation inputs or any confirmation outcomes and did not execute model computation. Blinding remains procedural on a shared filesystem. No forecast or gate may change after remote preregistration readback or outcome release.
