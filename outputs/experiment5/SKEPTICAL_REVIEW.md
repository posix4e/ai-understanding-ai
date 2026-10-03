# E5 independent skeptical outcome review

**The registered primary claim passes, and the evidence supports a bounded publication contribution.** A fixed, structure-informed attention repair transferred to six fresh models across the complete 105-layout class. The two registered boundary accuracy claims also pass. The separate all-six-model claim that each model's pooled edge-redirection effect has a lower confidence bound above .05 fails in seeds 7 and 10. This failure must remain part of the result.

## Independent audit and prospective integrity

The auditor ran no models and changed no frozen files. An independent program importing neither the model nor the official scorer checked all **64,272 saved condition cells**, representing **21,301,248 condition-row observations**, in **492 shards**. These observations share dictionaries and are not independent replications. Every expected layout/condition was present once. Full 16-class probabilities, normalization, predicted classes, target probabilities, accuracy, finite values, diagnostic signs, native-distribution discrepancy checks and all shard hashes passed.

All **96 frozen file hashes** still match the public-lock record for commit `de745644b5c71853755e8d53cc9b3dec2abdc2ed`. Recorded public verification occurred at **16:46:17.917703 UTC**, before confirmation started at **16:46:45.385492 UTC**; completion was **17:10:11.284599 UTC**. This audit checked the retained provenance and local hashes; the pre-run remote blob verification was performed by the preregistration procedure.

Independent row-stratified bootstrap calculations used the registered 2,000 draws and seeds, preserving pairing across layouts, controls and checkpoints. They reconstructed all **2,520 primary layout/query accuracy gates**, the nested 102-layout control estimand, native-deficit intervals, four boundary policies, pooled and individual edge intervals, and all 72 numerical forecasts. All decisions agree. The maximum discrepancy in audited aggregate means and principal intervals is **5.46e-8**, attributable to precision/averaging order. Individual edge intervals agree within 1e-7. Evidence is retained in [independent_audit.json](independent_audit.json).

Implementation checks pass: maximum correct-value/query state discrepancy is **1.91e-6**, maximum full-oracle probability discrepancy **2.38e-7**, and no-op distribution error, untouched-key error and boundary own-key/self reference-state error are zero. These are manipulation checks, not empirical discoveries.

## What the primary result establishes

Each model was tested on the same 1,024 new dictionaries, with 256 queries per pair index. No repair parameters were fitted to these models or layouts. The control family exhausts the allowed key masks that preserve own-key access and match row degrees.

| Seed | Primary accuracy | Worst layout/query accuracy | Upper 95% bound: native−guard P | Lower 95% bound: guard−mean shams P |
| --- | ---: | ---: | ---: | ---: |
| 6 | 1.000000 | 1.000000 | 0.00000388 | 0.09056 |
| 7 | 0.999860 | 0.996094 | 0.00062329 | 0.19779 |
| 8 | 1.000000 | 1.000000 | 0.00001263 | 0.10932 |
| 9 | 1.000000 | 1.000000 | 0.00002664 | 0.24845 |
| 10 | 1.000000 | 1.000000 | 0.00000398 | 0.03871 |
| 11 | 1.000000 | 1.000000 | 0.00005446 | 0.13842 |

The requirements were every layout/query accuracy ≥.95, native-deficit upper bound ≤.01, and mean-control advantage lower bound >.02, in every seed. All pass. Descriptively, unguarded accuracy over the same primary layouts ranges from **45.2% to 71.9%**, so this is a substantial behavioral repair rather than a tiny improvement to an already successful layout distribution.

There is no best-sham counterexample by mean target probability in this sample: the correct guard beats every individual sham within every eligible layout/model. The closest point-estimate advantage is **.00482** in seed 10. This is a descriptive check, not a simultaneous selected-best confidence claim. Controls match key counts and own-key retention; they do not match attention mass removed or activation perturbation size. The experiment does not show that the correct guard is unique or minimal.

### The repair is not exact

Seed 7 has **15 incorrect condition-row answers across 15 layouts**, arising from **two dictionary rows**, zero-based 576 and 770. There are no primary errors in the other five models. In the worst case, correct-answer probability falls from native **.999083** to guarded **.008674**, and another displayed value wins. This is a severe individual failure hidden by an excellent mean. All first-layer value states are nevertheless restored within tolerance, and the full-key oracle recovers native output. Thus the uncorrected key-token states remain a real route to failure. The empirical near-native claim passes; any claim of exact answer restoration is false. Individual cases remain in [primary_failure_cases.csv](primary_failure_cases.csv).

## Boundary success and what it does not imply

The remaining 2,415 layouts were tested on the frozen first 256 dictionaries, 64 per query position.

| Seed | Unguarded accuracy | Keyguard accuracy | Logical-prefix accuracy | Own-key/self accuracy |
| --- | ---: | ---: | ---: | ---: |
| 6 | .68577 | .99659 | 1.00000 | 1.00000 |
| 7 | .58062 | .99265 | 1.00000 | 1.00000 |
| 8 | .64069 | 1.00000 | 1.00000 | 1.00000 |
| 9 | .39569 | .99484 | 1.00000 | 1.00000 |
| 10 | .67517 | 1.00000 | 1.00000 | 1.00000 |
| 11 | .59268 | .99755 | 1.00000 | 1.00000 |

Logical-prefix and own-key/self each have zero observed errors across the boundary sample in all six seeds, including every layout/query cell. The simpler keyguard has counterexamples: its worst layout/query accuracy is **.890625** in seed 9, despite high pooled accuracy. The complete tables retain these tails.

These results show that exact reconstruction of the native predecessor set is stronger than needed for successful behavior on these sampled dictionaries. They do not extend the exact-restoration theorem to missing-context layouts. Own-key/self explicitly supplies the correct key/value association and yields a different, layout-invariant representation. Its success cannot establish that the model naturally discovers that invariant algorithm. The 256 shared dictionaries remain the data size; millions of repeated interventions do not increase it to millions of independent examples. Zero-error bootstrap accuracy intervals degenerate at one and are not guarantees of perfect population accuracy.

## Negative and imperfect predictions

The specified edge test re-adds one blocked key-to-value connection and measures the resulting increase in its named wrong value relative to the other two wrong displayed values. Equal-edge pooled effects and paired 95% intervals are:

| Seed | Edge effect [95% CI] | Lower bound >.05 |
| --- | --- | --- |
| 6 | .06000 [.05124, .07000] | pass |
| 7 | .04645 [.03813, .05539] | **fail** |
| 8 | .10471 [.09511, .11463] | pass |
| 9 | .21200 [.19344, .23024] | pass |
| 10 | .05313 [.04500, .06134] | **fail** |
| 11 | .08876 [.07600, .10238] | pass |

The effects are directionally positive in all models, but the preregistered magnitude claim fails. Individual edge point effects span roughly **.000064 to .439**. That heterogeneity argues against a uniform-strength interference account or a claim that every deleted edge is behaviorally necessary. The test examines reinsertion conditional on the other five deletions; it does not establish joint necessity or minimality.

Numerical forecasts have **65/72** outcomes within the frozen ±.05 tolerance, MAE **.02075**, RMSE **.03597**. Seven misses comprise mean-control advantages in seeds 7, 9 and 11, and edge magnitudes in seeds 6, 7, 9 and 10. The largest error is **.17938**, for seed 9's control advantage. These judgments correctly anticipated recovery but did not quantitatively capture how interference strength varies across models. Aggregate forecast success is not per-condition prediction accuracy.

## Publication judgment in plain language

The useful result is that a concrete explanation produced a repair that worked on new models and an exhaustively specified set of unfamiliar layouts, with quantitative predictions made public before testing. Changing the order of the dictionary had exposed tokens to misleading information. Restricting particular attention connections largely fixed the answers. The experiment also identifies where the explanation is incomplete: two dictionary counterexamples, residual key-state effects, and weaker-than-predicted edge interference in two models.

This is defensible as a carefully controlled, reproducible synthetic mechanistic study. The contribution combines a precise architectural guarantee, prospective transfer tests, a complete finite control family, and retained failures. The guarantee itself is elementary, and positional binding, attention masking and context restoration have prior literature. Field-wide novelty and publication acceptance are not established by this audit.

A suitable claim is: **“A fixed, association-informed visibility restriction prospectively transfers across six newly trained tiny transformers and the complete native-value-order serialization class, with near-native behavior and an advantage over matched masks; broader deletion policies also succeed on the registered boundary sample.”** Claims of autonomous understanding, a unique learned algorithm, ordinary position-independent generalization, arbitrary-dictionary certainty, or transfer to large pretrained language models would exceed the evidence. E3's failed position-only explanation should remain central to the account of how this narrower claim was reached.
