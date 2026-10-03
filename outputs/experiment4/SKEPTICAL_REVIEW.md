# Experiment 4 independent skeptical outcome review

**The complete preregistered E4 conjunction passes in all six models.** Removing the specified newly exposed first-layer key edges restores near-native retrieval, exceeds the required improvement over unguarded grouping, and beats the mean of the eight matched masks. This is an empirical partial-repair result. The exact restoration of first-layer value states and the full-state oracle are architectural identities, not discoveries.

E3 remains a failed prediction: its position-label sufficiency claim failed in every model, with 937/1,200 numeric forecast misses. E4 adds a separately registered intervention after that failure; it does not validate the earlier prediction retrospectively.

## Integrity and independent recomputation

All 44 E4 frozen files retain their registered hashes. Commit `b11245db8da95c4a98f9b25a452e828b7c3abe72` was recorded remotely verified at `2026-10-03T14:45:51.004522+00:00`; the matching confirmation started at `14:45:59.874058+00:00` and completed at `14:46:09.609038+00:00`. I checked the local lock and frozen hashes rather than performing a second remote-network readback. Blinding remains procedural, not OS-enforced.

I independently audited all 90 saved model/condition cells: finite normalized 16-class probabilities, matching argmax, correct target-class extraction, and consistent accuracies. All 180 forecast errors and every declared gate were recomputed. Independent integer multinomial counts within each of the four query strata reproduce all 18 requested intervals—guard minus grouped, guard minus mean eight, and guard minus reselected best control—within 1e-12. I also reproduced 450 overall/per-query Wilson intervals. The bootstrap preserves whole rows and uses shared draws across conditions and models.

The full-class guard/value-oracle discrepancy is at most 2.98e-7; the full-oracle/native discrepancy is at most 1.19e-7. Recorded first-layer value/query restoration errors are at most 1.91e-6, below the 1e-5 tolerance; key states remain exactly unchanged from unguarded grouping, and no-op target errors are zero. The saved state-error arrays were checked; full residual vectors were not stored for an independent reconstruction of those differences. Preflight tests independently verified the patch implementation on random weights. No new model forward was run for this review.

## Registered empirical result

Each model evaluated the same 2,048 fresh dictionaries, with 512 queries per pair position. All native accuracies are 100%. Individual model/query cells, rather than a pooled case count, determine the registered recovery decision.

| Seed | Grouped accuracy | Guard correct / 2,048 | Minimum query accuracy | Guard−grouped probability [95% CI] | Guard−mean eight controls [95% CI] |
| --- | --- | --- | --- | --- | --- |
| 0 | .5332 | 2,048 | 1.0000 | .4804 [.4648, .4964] | .1418 [.1336, .1503] |
| 1 | .5645 | 2,048 | 1.0000 | .4412 [.4280, .4547] | .1079 [.1024, .1138] |
| 2 | .2891 | 2,048 | 1.0000 | .7140 [.6975, .7288] | .3463 [.3376, .3543] |
| 3 | .3853 | 2,048 | 1.0000 | .6128 [.5998, .6252] | .1637 [.1563, .1710] |
| 4 | .3408 | 2,048 | 1.0000 | .6680 [.6512, .6850] | .2613 [.2522, .2704] |
| 5 | .5820 | 2,047 | .9980 | .4284 [.4143, .4438] | .1058 [.0984, .1133] |

Every query accuracy exceeds .95. Every guard-minus-grouped lower bound exceeds .30, and every guard-minus-average-control lower bound exceeds .05. These passes are not attributable to one model, one favorable query position or a post-hoc threshold change.

This supports a narrow causal statement: in these six fixed models and this grouped canonical layout, restoring training-time key visibility at first-layer value nodes is sufficient for near-native accuracy despite leaving first-layer key states altered. The matched family shows that edge identity matters beyond its mean sparsification effect. It does not establish that the prescribed mask is the unique sufficient repair.

## Counterexamples and non-uniqueness

Several alternative masks are strong repairs. In seed 1, both `guard_a0_b0` and `guard_a0_b1` achieve 2,047/2,048 correct answers and exceed .95 in every query stratum. In seed 5, `guard_a2_b3` achieves 2,032/2,048 and also exceeds .95 in every stratum. These are counterexamples to uniqueness, not failures of the registered mean-control comparison.

The correct guard has a positive point advantage over the best individual control in every model, but that advantage ranges from .000295 to .194855 in mean target probability. Seed 1's best-control difference is practically tiny; its descriptive reselected-bootstrap interval is approximately [.000007, .000847]. A positive interval does not make that difference substantively large. The experiment did not preregister a requirement to beat every sham by a meaningful amount.

There is also one direct counterexample to exact value-path sufficiency. In seed 5, input row 260 (zero-based, query index 0), the correct guard predicts class 14 instead of class 1: target probability is .1360, compared with .999536 under native and the full guard-plus-key oracle. Thus the remaining altered key states can matter decisively on an individual example even while the near-native accuracy criterion passes. This observation is based on a saved row, not a newly selected intervention.

Key-state restoration alone produces much smaller average probability changes than the guard: approximately −.0012 to +.0124 across seeds. After the guard, key restoration adds about .000001 to .000925 on average. These prespecified descriptive contrasts support the concentration of the mean deficit in the value pathway, while the single error cautions against treating the other states as irrelevant or strictly unnecessary.

## Forecast accuracy and its failures

**176/180 numerical forecasts** fall within the frozen ±.15 tolerance; overall RMSE is **.06715**. Separating easier references and oracles, **104/108 matched-guard forecasts** pass, with RMSE **.08631**. All four misses are seed 3's `guard_a3_b0` and `guard_a3_b1`, for probability and accuracy. These masks perform better than forecast: predicted probability about .6777 versus .9043/.9054, and predicted accuracy .6874 versus .9097/.9102. The largest error is .22775.

The failures limit the local-query mixture explanation used to forecast shams. Counting exactly restored value rows is not enough to predict every mask's behavior; the effect depends on which rows and competing states remain. High overall tolerance coverage should not be described as a complete quantitative account. Oracle and reference forecasts also should not inflate claims about predicting novel effects.

## Scope of the supported result

The useful addition to E3 is a prospectively confirmed failure boundary and partial causal repair: retaining position coordinates alone was insufficient, while restoring a specified subset of causal predecessor relationships repaired almost all outputs. The nontrivial observation is final behavioral recovery despite changed key-node states. The first-layer equality theorem supplies a controlled intervention, not the empirical conclusion itself.

This remains an external mask supplied with knowledge of the training format. It does not demonstrate spontaneous grouped-layout adaptation, necessity of every removed edge, a unique circuit, arbitrary-layout transfer, or a universal transformer mechanism. All masks share V0's forced sparsification, and the controls match row degrees rather than attention mass or activation norms. Attention summaries are descriptive. Intervals condition on the six reused checkpoints and sampled dictionaries; individual intervals are not simultaneous bands, and shared dictionaries do not create independent training replications.

The experiment provides a defensible result within this task and model family. Field-wide novelty is not established: positional binding, steering and attention interventions have relevant precedents. Any broader mechanism or transfer claim requires its own prospective test, while E3's failed forecasts and E4's counterexamples remain visible.
