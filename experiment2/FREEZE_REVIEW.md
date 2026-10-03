# Experiment 2 final prospective freeze review

**No critical pre-freeze blocker found.** The final seven-baseline protocol, numerical forecasts, scoring rules and phase gate are consistent. Confirmation must still wait for successful publication and exact remote-content verification. This review accessed implementation, existing discovery evidence and forecast files only. It did not run a trained model, inspect confirmatory outcomes, or modify any Experiment 1 artifact.

## Numerical reproducibility

Independently regenerated the full final `predictions.json` from `predictor.py` using only the condition specifications and six discovery NPZ files/summary. All **870 forecasts**, fitted parameters, source hashes and metadata exactly match. Each seed has all 145 expected condition IDs; probabilities are finite and in [0,1]. The predictor imports no model runner or Torch and needs no confirmation input.

Independently regenerated all **6,090 baseline forecasts**: six seeds × seven methods × 145 conditions. Every value exactly matches `baseline_predictions.json`, is finite and lies in [0,1]. The two case-logit baselines use the same paired discovery probabilities as the AI predictor; the other five use discovery condition means. No reserved pair, triple/rescue or cross-layer outcomes enter either forecast construction.

The AI rule allocates the observed all-head log-odds interaction residual across pairs according to singleton-strength products. The fixed positive floor ensures the normalizing denominator exists. This constrains the four-head endpoint but does not identify the real pair/triple response; the documents correctly present that allocation as a testable extrapolation. Cross-layer forecasts assume additive singleton log-odds changes.

**The AI and case-logit-additive baseline make the same predictions for all 32 cross-layer conditions**, up to floating-point differences of at most 1.12e-16 in every seed. Cross-layer accuracy can support or reject their shared assumption, but cannot establish added AI value over that baseline in that family. Any overall advantage over that method must come from within-layer predictions. This is a prospective mathematical equivalence, not an outcome-dependent observation.

## Scorer and decision rules

Reviewed `score.py` against the final protocol and prediction metadata:

* Primary MSE averages the three novel-family MSEs with weight 1/3 each: 36 pair, 24 triple/rescue and 32 cross-layer cells. All 53 familiar/reference/control cells are excluded. RMSE is the square root of this balanced MSE.
* Practical gates are RMSE <=0.10 and at least 80% of the 92 novel means within absolute error 0.10, equivalently at least 74. Each seed receives both gates separately.
* The bootstrap uses 2,000 multinomial case-count draws, seed 6200001. One weight matrix is shared across all conditions and all six models, preserving their common-case dependence.
* The comparator is the minimum balanced loss among all seven frozen baseline methods, recomputed within every paired resample. The superiority gate is an upper 95% percentile endpoint strictly below zero. This is distinct from both absolute-accuracy gates.
* Validity gates check clean accuracy >=95%, maximum runtime no-op target-probability error <=1e-6, and finite probabilities. The protocol distinguishes this runtime probability check from the synthetic preflight logit checks.
* `global_decisions.json` records all-six validity, all-six predictive adequacy, all-six baseline superiority and their conjunction. This global rule is an all-model conjunction, not an estimated population-average effect or an additional aggregate confidence interval.

No scoring arithmetic discrepancy was found. The matched bootstrap and frozen best-baseline comparison are stronger than selecting a convenient baseline afterward. Their uncertainty remains conditional on training, calibration, discovery fitting and the chosen forecast families. The bootstrap does not establish coverage across future training runs or compensate for every modeling decision made before registration.

The scorer reports descriptive scores even if validity or practical gates fail. A report must distinguish those descriptive gates: the existence of an `all_six_predictively_adequate` flag alone is not enough for a scientifically valid success claim if `all_six_valid` is false. The final protocol states that distinction correctly.

## Phase and integrity boundaries

`run_phase.py` requires the Experiment 2 lock before confirmation, verifies every frozen-file hash and the remote commit, refuses an existing result directory, then loads the frozen confirmation inputs and calibration tensors. Calibration is computed only in discovery; confirmation loads the saved means. The runner itself remains an ungated computation primitive, so the phase controller and procedural role separation are part of the experiment's trust boundary.

`preregister.py` includes Experiment 2 source, protocols, prediction files, all current Experiment 2 input/discovery/calibration/checkpoint/review artifacts, relevant shared source and existing checkpoints. It reads back the exact remote manifest and Git tree and checks every local blob identity before writing the separate Experiment 2 lock. It refuses retrospective registration after the confirmation directory exists. Successful verification must be recorded before any confirmation forward.

The earlier implementation/input review remains applicable: 92 novel interventions are distinct from discovery after canonicalizing rescue/complement equivalence; selected inputs are disjoint from the audited forbidden set and each other; live cross-layer recomputation passed direct-hook synthetic comparisons. This freeze review does not repeat those model/data tests or claim an independent new audit of every input.

At the end of this review, all **45 Experiment 1 frozen hashes remain unchanged**, and the Experiment 2 confirmatory result directory is absent. Existing and new seeds must remain separately visible in the final report, including failures. Passing this test would establish prospective utility of the complete frozen numerical rule on this grid, not unique mechanistic semantics or strict natural necessity.
