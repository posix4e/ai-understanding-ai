# Experiment 5: causal visibility across all restorable serializations

**The full prespecified six-seed primary claim passed.**

Recovery, control specificity, implementation validity, secondary findings, and numerical forecasts are separate results. An algebraic identity check or secondary success cannot rescue a primary failure.

| Decision | Result |
| --- | --- |
| all_six_valid | pass |
| all_six_recovery | pass |
| all_six_specificity | pass |
| primary_claim_pass | pass |

## Primary result: every fresh model

| Seed | Native accuracy | Correct-guard accuracy | Worst layout/query accuracy | Native−guard P [95% CI] | Guard−mean shams P [95% CI] | Failed validity/primary gates |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] | 0.0980 [0.0906, 0.1059] | none |
| 7 | 1.0000 | 0.9999 | 0.9961 | 0.0003 [0.0001, 0.0006] | 0.2064 [0.1978, 0.2148] | none |
| 8 | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] | 0.1175 [0.1093, 0.1259] | none |
| 9 | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] | 0.2594 [0.2484, 0.2706] | none |
| 10 | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] | 0.0460 [0.0387, 0.0537] | none |
| 11 | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0001] | 0.1502 [0.1384, 0.1618] | none |

Recovery requires ≥0.95 accuracy in every one of 105×4 layout/query cells per seed, and an upper paired 95% bound on native-minus-guard probability ≤0.01. Specificity requires the lower bound on guard-minus-mean-shams probability strictly >0.02. Every gate must pass in all six seeds. The native-minus-guard contrast is signed; it is a noninferiority criterion rather than a bound on absolute probability differences.

## Complete failures and controls

[Every wrong primary prediction](primary_failure_cases.csv), including mistakes within passing cells, [every failing primary layout/query cell](primary_failures.csv), [every condition mean](condition_results.csv), and [every query-stratified result](per_query_results.csv) are retained. The condition and query tables include all references, oracles, shams, boundary layouts, and edge tests; full softmax distributions remain in the raw shards.

| Seed | Failed primary cells | Worst layout | Worst query | Accuracy |
| --- | --- | --- | --- | --- |
| 6 | 0 | layout_64201357 | 3 | 1.0000 |
| 7 | 0 | layout_46201357 | 2 | 0.9961 |
| 8 | 0 | layout_24601357 | 1 | 1.0000 |
| 9 | 0 | layout_24601357 | 1 | 1.0000 |
| 10 | 0 | layout_60421357 | 3 | 1.0000 |
| 11 | 0 | layout_04621357 | 0 | 1.0000 |

[All control summaries](controls_summary.csv) show the correct mask, mean of each layout's complete sham family, and best individual sham by point estimate. There are 624 shams across 102 eligible layouts. Controls are averaged within each layout, then layouts receive equal weight. Three layouts admit no alternative that preserves the own key and row degree; they remain recovery tests but are excluded from specificity. Best-sham points are descriptive and have no selected-best confidence interval. Success against the mean does not establish superiority to every sham.

## Separate secondary results

| Secondary all-six-seed decision | Result |
| --- | --- |
| boundary_logical_prefix | pass |
| boundary_own_key_self | pass |
| edge_redirection | FAIL |

| Seed | Boundary prefix accuracy [95% CI] | Boundary own-key/self accuracy [95% CI] | Prefix cells ≥.95 | Own-key/self cells ≥.95 | Edge redirection [95% CI] |
| --- | --- | --- | --- | --- | --- |
| 6 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0600 [0.0512, 0.0700] |
| 7 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0465 [0.0381, 0.0554] |
| 8 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.1047 [0.0951, 0.1146] |
| 9 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.2120 [0.1934, 0.2302] |
| 10 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0531 [0.0450, 0.0613] |
| 11 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 0.0888 [0.0760, 0.1024] |

The two boundary accuracy gates require the pooled equal-layout accuracy ≥0.95 in every seed. The fraction of individual layout/query cells above .95 is descriptive, not an additional gate. Boundary queries have 64 cases per stratum. The edge gate requires a lower paired 95% bound strictly >0.05 in every seed.

| Seed | All 2,415 boundary layouts: policy | Target probability [95% CI] | Accuracy [95% CI] |
| --- | --- | --- | --- |
| 6 | unguarded | 0.6760 [0.6432, 0.7104] | 0.6858 [0.6506, 0.7216] |
| 6 | keyguard | 0.9956 [0.9913, 0.9988] | 0.9966 [0.9921, 0.9999] |
| 6 | logical_prefix | 0.9994 [0.9993, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 6 | own_key_self | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 7 | unguarded | 0.5632 [0.5298, 0.5963] | 0.5806 [0.5447, 0.6167] |
| 7 | keyguard | 0.9918 [0.9850, 0.9972] | 0.9927 [0.9855, 0.9983] |
| 7 | logical_prefix | 0.9991 [0.9988, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 7 | own_key_self | 0.9991 [0.9988, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 8 | unguarded | 0.6298 [0.6021, 0.6573] | 0.6407 [0.6113, 0.6695] |
| 8 | keyguard | 0.9993 [0.9993, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 8 | logical_prefix | 0.9994 [0.9993, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 8 | own_key_self | 0.9994 [0.9993, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 9 | unguarded | 0.3840 [0.3487, 0.4209] | 0.3957 [0.3574, 0.4357] |
| 9 | keyguard | 0.9935 [0.9885, 0.9975] | 0.9948 [0.9898, 0.9987] |
| 9 | logical_prefix | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 9 | own_key_self | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 10 | unguarded | 0.6747 [0.6446, 0.7045] | 0.6752 [0.6434, 0.7070] |
| 10 | keyguard | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 10 | logical_prefix | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 10 | own_key_self | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 11 | unguarded | 0.5799 [0.5423, 0.6152] | 0.5927 [0.5530, 0.6310] |
| 11 | keyguard | 0.9969 [0.9931, 0.9993] | 0.9975 [0.9936, 1.0000] |
| 11 | logical_prefix | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |
| 11 | own_key_self | 0.9994 [0.9994, 0.9994] | 1.0000 [1.0000, 1.0000] |

For re-added Vi←Kj edges, the edge statistic uses only queries for Kj. It measures the change in probability of the wrong value Vi relative to the mean change of the two other displayed wrong values, compared with the grouped correct guard. Each of six edges receives equal weight after within-query normalization. This comparison controls for nonspecific redistribution among wrong answers; q0 is not eligible for this aggregate.

| Seed | Edge | Eligible-query contrast [95% CI] |
| --- | --- | --- |
| 6 | edge_v0_k1 | 0.0476 [0.0306, 0.0666] |
| 6 | edge_v0_k2 | 0.0888 [0.0648, 0.1153] |
| 6 | edge_v0_k3 | 0.1785 [0.1385, 0.2223] |
| 6 | edge_v1_k2 | 0.0334 [0.0228, 0.0462] |
| 6 | edge_v1_k3 | 0.0113 [0.0031, 0.0220] |
| 6 | edge_v2_k3 | 0.0005 [0.0002, 0.0009] |
| 7 | edge_v0_k1 | 0.0256 [0.0114, 0.0430] |
| 7 | edge_v0_k2 | 0.0496 [0.0286, 0.0729] |
| 7 | edge_v0_k3 | 0.0598 [0.0391, 0.0833] |
| 7 | edge_v1_k2 | 0.0013 [0.0002, 0.0032] |
| 7 | edge_v1_k3 | 0.0394 [0.0225, 0.0585] |
| 7 | edge_v2_k3 | 0.1030 [0.0811, 0.1282] |
| 8 | edge_v0_k1 | 0.0610 [0.0411, 0.0821] |
| 8 | edge_v0_k2 | 0.0521 [0.0311, 0.0765] |
| 8 | edge_v0_k3 | 0.4388 [0.3972, 0.4811] |
| 8 | edge_v1_k2 | 0.0145 [0.0073, 0.0233] |
| 8 | edge_v1_k3 | 0.0592 [0.0403, 0.0812] |
| 8 | edge_v2_k3 | 0.0027 [0.0003, 0.0070] |
| 9 | edge_v0_k1 | 0.1618 [0.1252, 0.2008] |
| 9 | edge_v0_k2 | 0.2695 [0.2235, 0.3183] |
| 9 | edge_v0_k3 | 0.3560 [0.3043, 0.4096] |
| 9 | edge_v1_k2 | 0.0917 [0.0621, 0.1240] |
| 9 | edge_v1_k3 | 0.3764 [0.3230, 0.4283] |
| 9 | edge_v2_k3 | 0.0166 [0.0089, 0.0260] |
| 10 | edge_v0_k1 | 0.0226 [0.0111, 0.0376] |
| 10 | edge_v0_k2 | 0.2386 [0.1984, 0.2822] |
| 10 | edge_v0_k3 | 0.0474 [0.0305, 0.0691] |
| 10 | edge_v1_k2 | 0.0100 [0.0041, 0.0183] |
| 10 | edge_v1_k3 | 0.0001 [0.0000, 0.0001] |
| 10 | edge_v2_k3 | 0.0001 [0.0001, 0.0001] |
| 11 | edge_v0_k1 | 0.0591 [0.0373, 0.0839] |
| 11 | edge_v0_k2 | 0.0539 [0.0326, 0.0790] |
| 11 | edge_v0_k3 | 0.2871 [0.2397, 0.3332] |
| 11 | edge_v1_k2 | 0.0305 [0.0175, 0.0461] |
| 11 | edge_v1_k3 | 0.0612 [0.0367, 0.0882] |
| 11 | edge_v2_k3 | 0.0408 [0.0225, 0.0615] |

### Fixed key order and permuted values

The fixed-key-order subpanel consists of 23 boundary value orders plus the identity value order from primary grouping. They are shown separately below; no mixed-sample or missing-condition 24-layout average is manufactured. Primary grouped references here use the same first 256 rows as the boundary panel.

| Seed | 23 boundary layouts: unguarded accuracy | 23: keyguard accuracy | 23: logical-prefix accuracy | 23: own-key/self accuracy | Grouped identity: unguarded | Grouped identity: correct guard |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.6406 | 0.9949 | 1.0000 | 1.0000 | 0.6406 | 1.0000 |
| 7 | 0.4946 | 0.9912 | 1.0000 | 1.0000 | 0.4961 | 1.0000 |
| 8 | 0.6213 | 1.0000 | 1.0000 | 1.0000 | 0.6250 | 1.0000 |
| 9 | 0.3478 | 0.9891 | 1.0000 | 1.0000 | 0.3438 | 1.0000 |
| 10 | 0.6445 | 1.0000 | 1.0000 | 1.0000 | 0.6445 | 1.0000 |
| 11 | 0.5338 | 0.9978 | 1.0000 | 1.0000 | 0.5312 | 1.0000 |

## Validity and numerical predictions

| Seed | Validity | No-op maximum error | Correct value-state error | Query-state error, all cells | Untouched-key error | Full-oracle probability error | Own-key/self value-state error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | pass | 0 | 1.43e-06 | 1.91e-06 | 0 | 2.38e-07 | 0 |
| 7 | pass | 0 | 1.91e-06 | 1.19e-06 | 0 | 1.19e-07 | 0 |
| 8 | pass | 0 | 1.43e-06 | 1.67e-06 | 0 | 1.19e-07 | 0 |
| 9 | pass | 0 | 1.91e-06 | 1.19e-06 | 0 | 1.19e-07 | 0 |
| 10 | pass | 0 | 1.67e-06 | 1.31e-06 | 0 | 1.19e-07 | 0 |
| 11 | pass | 0 | 1.67e-06 | 1.91e-06 | 0 | 1.19e-07 | 0 |

All saved output cells are checked for complete shapes, finite values, normalized full-class probabilities, argmax/accuracy/target consistency, and nonnegative state errors. Shard hashes and layout coverage are checked against completion ledgers. No-op distribution tolerance is 1e-6; registered state/oracle tolerances are 1e-5. Individual state differences are measured by the frozen runner; independent synthetic tests audit the hook implementation.

The separate 72 aggregate numerical forecasts have MAE **0.0208**, RMSE **0.0360**, and **65/72** outcomes within ±0.05. [Every forecast](forecasts_vs_results.csv) and [every miss](forecast_misses.csv) are included. These are judgmental point forecasts, not predictive intervals, and their scores do not replace any scientific gate.

## Interpretation and limits

All 2,520 pair-respecting orders keep Ki before Vi. Exactly 105 preserve value order and therefore retain every native predecessor at every value node. With canonical IDs, removing extra future-logical keys must restore first-layer value states. Query states are invariant because the final query sees every token. Key states remain changed; their second-layer keys/values and attention normalization can still alter the answer. Thus final-answer recovery and the comparison with matched masks are empirical, while value-state and full-oracle restoration are algebraic manipulation checks. The exact-restoration proof is in [THEORY.md](THEORY.md).

The 2,415 remaining orders can omit earlier native keys or values. Deleting future keys alone, deleting all newly visible future-logical identities, and retaining only the own key plus self are distinct policies. Own-key/self gives invariant first-layer value states under all layouts, but those are not generally native states. These masks use known canonical pair identities; success does not demonstrate independent discovery of pair associations.

This is an exhaustive finite synthetic study, conditional on a fixed two-layer, four-head, 70,720-parameter architecture and six fresh training seeds. Canonical position IDs preserve training-slot information. The study does not establish ordinary position-invariant generalization, performance on naturally pretrained LLMs, minimal circuits, or a new general context-restoration principle. E1–E4, including negative results, remain preserved.

Confidence intervals use 2,000 query-stratified paired dictionary-row resamples, shared across layouts, conditions and seeds within panels. They describe uncertainty over dictionaries conditional on these checkpoints, not uncertainty over model training. Wilson intervals in CSVs are individual intervals, not simultaneous coverage. Layouts and interventions on the same dictionaries are dependent. The boundary uses the first 256 primary dictionaries; a separate bootstrap seed does not make it an independent dataset.

## Prospective record and reproduction

Frozen registration: [de745644b5c71853755e8d53cc9b3dec2abdc2ed](https://github.com/posix4e/ai-understanding-ai/commit/de745644b5c71853755e8d53cc9b3dec2abdc2ed). Public verification: `2026-10-03T16:46:17.917703+00:00`. Confirmation start: `2026-10-03T16:46:45.385492+00:00`; finish: `2026-10-03T17:10:11.284599+00:00`. Report generated: `2026-10-03T17:10:58.325831+00:00`.

Training seeds 6–11 and the unchanged 2,000-step recipe were publicly fixed before training. New inputs contain 1,024 distinct dictionaries with 256 queries per pair; the input audit checks historical exclusions. All inference ran locally on CPU with four threads; no inference API or new shifted-layout fitting was used. Raw shards, per-seed completion ledgers, [input audit](input_audit.json), [training cohort](training_cohort.json), and the machine-readable [scores](confirmatory/scores.json) retain provenance, counts, and failures.

To reproduce on a separate checkout, first preserve the published outcome directory and its hashes. The frozen model-evaluation command refuses to overwrite completed confirmation; interrupted execution uses verified shards only. Pure postprocessing can be reproduced from copied raw artifacts with:

```sh
.venv/bin/python -m experiment5.scorer
.venv/bin/python -m experiment5.report
```

Original model evaluation command: `.venv/bin/python -m experiment5.run_phase`; deterministic crash resumption: `.venv/bin/python -m experiment5.run_phase --resume`. Neither command is executed by this report.
