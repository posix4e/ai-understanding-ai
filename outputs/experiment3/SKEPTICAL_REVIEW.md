# Experiment 3 independent skeptical outcome review

**The registered strong rebinding claim failed in every model.** All 54 primary value-only cells missed 90% named-answer accuracy, all 54 matched coherent cells missed 95% original-answer accuracy, and every model failed the probability-margin requirement. These are valid negative outcomes: all six native accuracies were 100%, no-op errors were exactly zero, and the raw arrays were complete and finite. The result must not be renamed a successful primary experiment because weaker directional effects exist.

## Integrity and independent checks

The 42 frozen files retain their registered hashes. The lock names commit `42fd4236ecfa0535dacf9ddf6c824c41bd66f02e`, with remote contents verified at `2026-10-03T14:25:09.329884+00:00`. The matching confirmation start is `14:28:14.516699+00:00`; completion is `14:28:51.709058+00:00`. Thus the recorded remote verification precedes the trained forwards. This review checked the local lock and hashes; it did not perform a second remote-network readback or establish OS-enforced blinding.

Using only saved arrays, I checked all 300 model/condition cells: the 16-class probabilities normalize, their argmax matches the saved class, original and inverse-permutation target probabilities match the correct class entries, accuracies match targets, and attention arrays are finite and correctly shaped. I independently reconstructed all 1,200 forecast errors and all primary gates. An independent implementation generated integer multinomial counts separately within each query stratum and accumulated the nine-condition row means; all six primary probability confidence intervals agree with the scorer within 1e-12. The same audit reproduced the matched coherent wrong-target intervals and 600 individual accuracy Wilson intervals. No model forward or new intervention was run for this review.

## Size of the failure

Ranges below span the nine primary permutations, not confidence intervals. The last column is the paired query-stratified 95% interval for the mean slot-minus-original probability contrast.

| Seed | Value-only named-answer accuracy range | Coherent original accuracy range | Probability contrast [95% CI] |
| --- | --- | --- | --- |
| 0 | .535–.553 | .495–.537 | .435 [.418, .452] |
| 1 | .575–.587 | .564–.592 | .468 [.452, .483] |
| 2 | .273–.292 | .290–.340 | .107 [.088, .125] |
| 3 | .382–.393 | .380–.416 | .212 [.198, .227] |
| 4 | .328–.341 | .303–.355 | .171 [.154, .187] |
| 5 | .581–.589 | .555–.613 | .487 [.472, .500] |

The required contrast lower bound was greater than .80. None is close. The failure is not driven by one difficult permutation or one weak seed.

Only **263/1,200 numeric forecasts** fall within the frozen ±.15 tolerance; **937 miss**, with overall RMSE **.4485**. All **552 coherent-condition scalar forecasts** miss. Of 552 value-only forecasts, 361 miss; 24/96 reference forecasts miss, corresponding to the canonical-grouped forecasts. The worst absolute error is .6971: seed 2 slot accuracy was .27295 against the .97 forecast for `value_1032` and `value_2103`. These are substantial errors, not a dispute about a marginal statistical threshold.

## Supported boundary and weaker observations

The cleanest boundary is **canonical coordinates are insufficient to restore grouped-layout competence**. Canonical-grouped original accuracy is .5391, .5884, .2886, .3848, .3315 and .5845 across seeds, compared with 1.0 in the native layout. The six seeds all show partial improvement over native-grouped accuracy (.1929–.2085), but none approaches the predicted .97. The matched coherent permutations also fail, so the result cannot be explained solely as deliberate answer redirection under conflicting labels.

There is nevertheless selective positional influence. The prescribed wrong target gains probability relative to the same target under its coherent match: pooled paired differences are .128–.481, with positive individual 95% intervals in all six models. The prespecified secondary inverse-versus-forward comparison favors the inverse rule by .116–.500 in answer fidelity, again with positive individual intervals. These support a weaker causal influence of assigned coordinates. They do not satisfy the failed sufficiency claim, establish a complete algorithm, or validate the numerical forecasts.

The prespecified query breakdown exposes substantial heterogeneity. For example, seed 3's mean probability contrast is .979 for query pair 3 but −.140 for pair 2; seed 2's pair-1 contrast is −.061. Highlighting these particular cells is an interpretation after observing outcomes. They motivate a new hypothesis, not a retrospectively selected successful cohort. Broadly, differences among query positions and model seeds are much larger than variation among the nine derangements within a seed.

Attention diagnostics are consistent with disrupted binding: native L1 true-key attention is often high, while canonical grouping substantially lowers it, and value-only attention partly tracks the assigned slot. These are descriptive observations, not a causal mediation analysis. They do not establish which changed predecessors or competing nodes caused the errors.

## Limits and next-test discipline

Grouping alters causal predecessor sets even when each token retains its original trained position embedding. The current experiment therefore falsifies position-label sufficiency; it does not establish physical adjacency as the unique remaining explanation. An intervention on newly exposed attention edges is a plausible next test, but it must use a new frozen protocol, suitable edge-count controls, and explicit failure criteria. Its results cannot repair or overwrite this experiment's failed forecasts.

This is a systematic prospective test of an existing positional-binding idea, not evidence that positional steering was newly discovered. All confidence intervals condition on the six reused checkpoints and selected task distribution. Individual intervals are not simultaneous bands, and the six checkpoints do not justify a population-wide claim. Externally supplied embeddings are assisted interventions, not spontaneous layout transfer or general AI self-understanding.
