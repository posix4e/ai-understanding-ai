# Experiment 2: predicting attention-head interventions

The full prespecified all-six-seed claim passed.

| Global decision | Passed |
| --- | --- |
| all_six_valid | yes |
| all_six_predictively_adequate | yes |
| all_six_beat_best_baseline | yes |
| full_claim_pass | yes |

Predictive adequacy and superiority to baselines are separate claims. Failing superiority alone does not mean the forecasts missed the registered accuracy tolerances. A failed validity gate prevents the full scientific claim regardless of numerical forecast performance.

The primary endpoint is mean original-target probability under a frozen intervention. This experiment tests whether discovery-only forecasts predict unseen combinations of head ablations and rescues. It does not equate an ablation effect with strict necessity during natural computation.

Public preregistration: [f4b3f1d73952e4fa449b2d4bf1f139769b0d75ad](https://github.com/posix4e/ai-understanding-ai/commit/f4b3f1d73952e4fa449b2d4bf1f139769b0d75ad). Remote contents verified at `2026-10-03T14:00:33.670448+00:00`; confirmation started at `2026-10-03T14:00:49.683873+00:00`. This report was generated at `2026-10-03T14:01:39.918654+00:00`.

## Design and scope

Six locally trained CPU transformers have 70,720 parameters each, two layers, four heads per layer, width 64, and MLP width 128. Seeds 0–2 reuse the Experiment 1 models; seeds 3–5 are newly trained replications. The task presents four random key–value associations and queries one key. There are 16 possible value classes (uniform chance 6.25%); choosing randomly among the four displayed values gives 25%. The same frozen 2,048 confirmation cases are evaluated in each model.

The intervention site is each head's weighted value output before concatenation and output projection. Layer 1 interventions act at all four value positions (1, 3, 5, 7); Layer 2 interventions act at query position 8. Means are position-specific and frozen from 512 independent calibration cases. Discovery uses 1,024 cases and singleton/all-head conditions. Confirmation contains 145 conditions, including 92 novel ones: 36 pair ablations, 24 sole-head rescues (equivalent to triple-head corruptions), and 32 cross-layer compositions. Equivalent rescue/triple conditions are scored once. Cross-layer interventions retain the naturally recomputed unselected downstream heads.

The three novel families receive equal weight in MSE. RMSE is the square root of that balanced MSE. All six seeds must individually meet all three forecast gates: balanced RMSE ≤0.10; at least 74/92 novel errors within 0.10; and an upper 95% paired-bootstrap bound strictly below zero for AI MSE minus the best baseline MSE. Familiar conditions and controls are descriptive, not primary scored evidence.

Each seed also requires clean accuracy ≥95%, maximum no-op probability error ≤10⁻⁶, and finite probabilities. The full claim requires all six seeds to pass validity, predictive adequacy, and baseline superiority.

## Every seed and every gate

| Seed | Cohort | Clean accuracy | AI RMSE | Within 0.10 | AI−best MSE | Paired 95% interval | Failed forecast gates | Failed validity gates | Max no-op probability error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | reused | 1.0000 | 0.0525 | 84/92 (91.3%) | -0.01541 | [-0.01610, -0.01464] | none | none | 0 |
| 1 | reused | 1.0000 | 0.0682 | 83/92 (90.2%) | -0.00682 | [-0.00763, -0.00601] | none | none | 0 |
| 2 | reused | 1.0000 | 0.0746 | 78/92 (84.8%) | -0.00988 | [-0.01054, -0.00917] | none | none | 0 |
| 3 | new | 1.0000 | 0.0556 | 85/92 (92.4%) | -0.01248 | [-0.01307, -0.01190] | none | none | 0 |
| 4 | new | 1.0000 | 0.0422 | 88/92 (95.7%) | -0.01550 | [-0.01609, -0.01467] | none | none | 0 |
| 5 | new | 1.0000 | 0.0593 | 84/92 (91.3%) | -0.01338 | [-0.01409, -0.01269] | none | none | 0 |

The no-op error is the maximum per-case target-probability difference across both layers. A zero value supports implementation identity for these tested inputs; it is not an independent forecast success. All recorded gate decisions above are taken directly from the frozen scoring procedure.

## Comparison with all 7 baselines

| Seed | Method | Balanced MSE | Balanced RMSE | 95% bootstrap MSE interval |
| --- | --- | --- | --- | --- |
| 0 | AI forecast | 0.00275 | 0.0525 | [0.00255, 0.00309] |
| 0 | Zero effect | 0.12191 | 0.3492 | [0.11962, 0.12421] |
| 0 | Additive | 0.04052 | 0.2013 | [0.03915, 0.04197] |
| 0 | Multiplicative | 0.04527 | 0.2128 | [0.04387, 0.04674] |
| 0 | Head-count linear | 0.02531 | 0.1591 | [0.02446, 0.02628] |
| 0 | Head-count logit | 0.02246 | 0.1499 | [0.02160, 0.02344] |
| 0 | Per-case logit additive | 0.03164 | 0.1779 | [0.03037, 0.03299] |
| 0 | Per-case logit head-count | 0.01817 | 0.1348 | [0.01763, 0.01876] |
| 1 | AI forecast | 0.00465 | 0.0682 | [0.00430, 0.00512] |
| 1 | Zero effect | 0.11287 | 0.3360 | [0.11080, 0.11503] |
| 1 | Additive | 0.04307 | 0.2075 | [0.04154, 0.04464] |
| 1 | Multiplicative | 0.04696 | 0.2167 | [0.04542, 0.04854] |
| 1 | Head-count linear | 0.01827 | 0.1352 | [0.01756, 0.01911] |
| 1 | Head-count logit | 0.01538 | 0.1240 | [0.01478, 0.01612] |
| 1 | Per-case logit additive | 0.03077 | 0.1754 | [0.02935, 0.03221] |
| 1 | Per-case logit head-count | 0.01147 | 0.1071 | [0.01105, 0.01198] |
| 2 | AI forecast | 0.00557 | 0.0746 | [0.00524, 0.00600] |
| 2 | Zero effect | 0.12395 | 0.3521 | [0.12173, 0.12638] |
| 2 | Additive | 0.03144 | 0.1773 | [0.03035, 0.03270] |
| 2 | Multiplicative | 0.03571 | 0.1890 | [0.03460, 0.03701] |
| 2 | Head-count linear | 0.02283 | 0.1511 | [0.02214, 0.02363] |
| 2 | Head-count logit | 0.01883 | 0.1372 | [0.01817, 0.01955] |
| 2 | Per-case logit additive | 0.02355 | 0.1534 | [0.02255, 0.02471] |
| 2 | Per-case logit head-count | 0.01545 | 0.1243 | [0.01501, 0.01598] |
| 3 | AI forecast | 0.00310 | 0.0556 | [0.00290, 0.00342] |
| 3 | Zero effect | 0.13085 | 0.3617 | [0.12784, 0.13396] |
| 3 | Additive | 0.03238 | 0.1800 | [0.03128, 0.03363] |
| 3 | Multiplicative | 0.03608 | 0.1899 | [0.03495, 0.03735] |
| 3 | Head-count linear | 0.02376 | 0.1541 | [0.02301, 0.02469] |
| 3 | Head-count logit | 0.02781 | 0.1668 | [0.02648, 0.02935] |
| 3 | Per-case logit additive | 0.02071 | 0.1439 | [0.01973, 0.02184] |
| 3 | Per-case logit head-count | 0.01558 | 0.1248 | [0.01512, 0.01619] |
| 4 | AI forecast | 0.00178 | 0.0422 | [0.00161, 0.00209] |
| 4 | Zero effect | 0.18354 | 0.4284 | [0.17974, 0.18773] |
| 4 | Additive | 0.02796 | 0.1672 | [0.02711, 0.02887] |
| 4 | Multiplicative | 0.03137 | 0.1771 | [0.03050, 0.03230] |
| 4 | Head-count linear | 0.02757 | 0.1660 | [0.02673, 0.02854] |
| 4 | Head-count logit | 0.04482 | 0.2117 | [0.04311, 0.04675] |
| 4 | Per-case logit additive | 0.01729 | 0.1315 | [0.01652, 0.01813] |
| 4 | Per-case logit head-count | 0.01758 | 0.1326 | [0.01708, 0.01822] |
| 5 | AI forecast | 0.00352 | 0.0593 | [0.00329, 0.00386] |
| 5 | Zero effect | 0.14571 | 0.3817 | [0.14325, 0.14842] |
| 5 | Additive | 0.03531 | 0.1879 | [0.03426, 0.03652] |
| 5 | Multiplicative | 0.04002 | 0.2000 | [0.03894, 0.04127] |
| 5 | Head-count linear | 0.02075 | 0.1441 | [0.02008, 0.02148] |
| 5 | Head-count logit | 0.02522 | 0.1588 | [0.02415, 0.02646] |
| 5 | Per-case logit additive | 0.02487 | 0.1577 | [0.02394, 0.02593] |
| 5 | Per-case logit head-count | 0.01690 | 0.1300 | [0.01641, 0.01750] |

The comparison interval uses AI loss minus the minimum of the 7 baseline losses within each paired resample; the selected baseline may change across resamples. Point-estimate best baselines are seed 0: Per-case logit head-count; seed 1: Per-case logit head-count; seed 2: Per-case logit head-count; seed 3: Per-case logit head-count; seed 4: Per-case logit additive; seed 5: Per-case logit head-count.

The per-case logit additive and per-case logit head-count baselines use paired raw discovery probabilities, matching the AI forecast's access to case-level discovery information. They were fixed before confirmation.

### Reused versus newly trained seeds

| Cohort | Method | Mean seed MSE | Square root of mean seed MSE |
| --- | --- | --- | --- |
| Reused seeds 0–2 | AI forecast | 0.00433 | 0.0658 |
| Reused seeds 0–2 | Zero effect | 0.11958 | 0.3458 |
| Reused seeds 0–2 | Additive | 0.03835 | 0.1958 |
| Reused seeds 0–2 | Multiplicative | 0.04265 | 0.2065 |
| Reused seeds 0–2 | Head-count linear | 0.02214 | 0.1488 |
| Reused seeds 0–2 | Head-count logit | 0.01889 | 0.1374 |
| Reused seeds 0–2 | Per-case logit additive | 0.02865 | 0.1693 |
| Reused seeds 0–2 | Per-case logit head-count | 0.01503 | 0.1226 |
| New seeds 3–5 | AI forecast | 0.00280 | 0.0529 |
| New seeds 3–5 | Zero effect | 0.15337 | 0.3916 |
| New seeds 3–5 | Additive | 0.03188 | 0.1786 |
| New seeds 3–5 | Multiplicative | 0.03582 | 0.1893 |
| New seeds 3–5 | Head-count linear | 0.02403 | 0.1550 |
| New seeds 3–5 | Head-count logit | 0.03262 | 0.1806 |
| New seeds 3–5 | Per-case logit additive | 0.02095 | 0.1448 |
| New seeds 3–5 | Per-case logit head-count | 0.01669 | 0.1292 |
| All six | AI forecast | 0.00356 | 0.0597 |
| All six | Zero effect | 0.13647 | 0.3694 |
| All six | Additive | 0.03512 | 0.1874 |
| All six | Multiplicative | 0.03923 | 0.1981 |
| All six | Head-count linear | 0.02308 | 0.1519 |
| All six | Head-count logit | 0.02575 | 0.1605 |
| All six | Per-case logit additive | 0.02480 | 0.1575 |
| All six | Per-case logit head-count | 0.01586 | 0.1259 |

Cohort summaries are descriptive equal-seed averages, not additional pooled hypothesis tests. A favorable cohort average does not override a failed seed or satisfy the all-six primary rule.

## Where the forecasts failed or succeeded

| Seed | Novel family | AI RMSE | Best point baseline | Baseline RMSE | Family RMSE >0.10 |
| --- | --- | --- | --- | --- | --- |
| 0 | pair | 0.0602 | Per-case logit head-count | 0.1136 | no |
| 0 | triple_rescue | 0.0679 | Head-count logit | 0.1801 | no |
| 0 | cross_layer | 0.0053 | Per-case logit additive | 0.0053 | no |
| 1 | pair | 0.0583 | Per-case logit head-count | 0.0850 | no |
| 1 | triple_rescue | 0.1024 | Head-count logit | 0.1349 | yes |
| 1 | cross_layer | 0.0091 | Per-case logit additive | 0.0091 | no |
| 2 | pair | 0.0763 | Per-case logit head-count | 0.0910 | no |
| 2 | triple_rescue | 0.1041 | Head-count logit | 0.1606 | yes |
| 2 | cross_layer | 0.0082 | Per-case logit additive | 0.0082 | no |
| 3 | pair | 0.0550 | Per-case logit head-count | 0.0886 | no |
| 3 | triple_rescue | 0.0788 | Head-count logit | 0.1963 | no |
| 3 | cross_layer | 0.0075 | Per-case logit additive | 0.0075 | no |
| 4 | pair | 0.0435 | Per-case logit additive | 0.0798 | no |
| 4 | triple_rescue | 0.0534 | Per-case logit head-count | 0.2036 | no |
| 4 | cross_layer | 0.0247 | Head-count logit | 0.0114 | no |
| 5 | pair | 0.0690 | Per-case logit additive | 0.0975 | no |
| 5 | triple_rescue | 0.0745 | Head-count logit | 0.1858 | no |
| 5 | cross_layer | 0.0157 | Head-count logit | 0.0120 | no |

Family rows reveal localized failures; the 0.10 family flag is descriptive rather than an additional registered gate. No family is dropped from the primary score.

Largest novel error in each seed:

| Seed | Condition | Forecast | Observed | Observed mean 95% interval | Absolute error |
| --- | --- | --- | --- | --- | --- |
| 0 | L1_mean_pair03 | 0.2025 | 0.4492 | [0.43094, 0.46737] | 0.2466 |
| 1 | L1_zero_rescue0 | 0.9445 | 0.7280 | [0.71031, 0.74565] | 0.2166 |
| 2 | L1_zero_pair01 | 0.2989 | 0.5515 | [0.53186, 0.57117] | 0.2526 |
| 3 | L2_mean_rescue0 | 0.4233 | 0.6226 | [0.60467, 0.64055] | 0.1993 |
| 4 | L2_mean_rescue3 | 0.4289 | 0.5544 | [0.53678, 0.57196] | 0.1255 |
| 5 | L1_mean_pair12 | 0.2614 | 0.5140 | [0.49504, 0.53299] | 0.2526 |

Largest 12 errors across all novel seed–condition combinations:

| Seed | Condition | Forecast | Observed | Observed mean 95% interval | Absolute error |
| --- | --- | --- | --- | --- | --- |
| 5 | L1_mean_pair12 | 0.2614 | 0.5140 | [0.49504, 0.53299] | 0.2526 |
| 2 | L1_zero_pair01 | 0.2989 | 0.5515 | [0.53186, 0.57117] | 0.2526 |
| 0 | L1_mean_pair03 | 0.2025 | 0.4492 | [0.43094, 0.46737] | 0.2466 |
| 2 | L1_mean_pair01 | 0.3891 | 0.6162 | [0.59748, 0.63490] | 0.2271 |
| 5 | L1_mean_rescue3 | 0.2583 | 0.4817 | [0.46278, 0.50053] | 0.2233 |
| 2 | L1_zero_rescue2 | 0.2895 | 0.5126 | [0.49297, 0.53231] | 0.2231 |
| 1 | L1_zero_rescue0 | 0.9445 | 0.7280 | [0.71031, 0.74565] | 0.2166 |
| 1 | L1_zero_rescue1 | 0.7855 | 0.5815 | [0.56185, 0.60115] | 0.2040 |
| 3 | L2_mean_rescue0 | 0.4233 | 0.6226 | [0.60467, 0.64055] | 0.1993 |
| 2 | L1_mean_rescue2 | 0.3870 | 0.5826 | [0.56366, 0.60161] | 0.1957 |
| 5 | L1_zero_pair12 | 0.2678 | 0.4496 | [0.43031, 0.46897] | 0.1819 |
| 2 | L1_zero_rescue0 | 0.7545 | 0.5772 | [0.55761, 0.59675] | 0.1773 |

## Descriptive head effects

Each cell reports target probability / task accuracy on confirmation cases. Head indices are 0–3. These singleton/all-head conditions were available during discovery and are not novel successes.

| Seed | Layer | Corruption | Head 0 | Head 1 | Head 2 | Head 3 | All heads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | mean | 0.9214 / 0.9331 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9631 / 0.9712 | 0.1826 / 0.1914 |
| 0 | 1 | zero | 0.8930 / 0.9033 | 0.9992 / 1.0000 | 0.9992 / 1.0000 | 0.9352 / 0.9438 | 0.1788 / 0.1851 |
| 0 | 1 | resample | 0.9981 / 0.9995 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9983 / 0.9995 | 0.9993 / 1.0000 |
| 0 | 2 | mean | 0.9295 / 0.9580 | 0.9219 / 0.9370 | 0.9760 / 0.9946 | 0.9608 / 0.9629 | 0.1119 / 0.1226 |
| 0 | 2 | zero | 0.9306 / 0.9541 | 0.9181 / 0.9316 | 0.9770 / 0.9956 | 0.9661 / 0.9736 | 0.1098 / 0.1235 |
| 0 | 2 | resample | 0.8249 / 0.8369 | 0.7855 / 0.7983 | 0.8303 / 0.8452 | 0.9159 / 0.9229 | 0.0001 / 0.0000 |
| 1 | 1 | mean | 0.9528 / 0.9639 | 0.9943 / 0.9976 | 0.9980 / 1.0000 | 0.9992 / 1.0000 | 0.1833 / 0.1885 |
| 1 | 1 | zero | 0.9296 / 0.9409 | 0.9841 / 0.9883 | 0.9928 / 0.9961 | 0.9991 / 1.0000 | 0.2083 / 0.2104 |
| 1 | 1 | resample | 0.9979 / 0.9990 | 0.9991 / 1.0000 | 0.9986 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 |
| 1 | 2 | mean | 0.9559 / 0.9790 | 0.9913 / 0.9980 | 0.9431 / 0.9761 | 0.9616 / 0.9858 | 0.1129 / 0.1304 |
| 1 | 2 | zero | 0.9615 / 0.9858 | 0.9916 / 0.9980 | 0.9368 / 0.9688 | 0.9646 / 0.9922 | 0.1169 / 0.1426 |
| 1 | 2 | resample | 0.8507 / 0.8770 | 0.9582 / 0.9697 | 0.7805 / 0.7920 | 0.7815 / 0.8027 | 0.0001 / 0.0000 |
| 2 | 1 | mean | 0.9864 / 0.9912 | 0.8513 / 0.8633 | 0.9974 / 0.9995 | 0.9994 / 1.0000 | 0.1754 / 0.1807 |
| 2 | 1 | zero | 0.9562 / 0.9639 | 0.8115 / 0.8208 | 0.9926 / 0.9961 | 0.9993 / 1.0000 | 0.1857 / 0.1924 |
| 2 | 1 | resample | 0.9988 / 1.0000 | 0.9971 / 0.9985 | 0.9988 / 0.9995 | 0.9994 / 1.0000 | 0.9994 / 1.0000 |
| 2 | 2 | mean | 0.9794 / 0.9946 | 0.9939 / 1.0000 | 0.9341 / 0.9536 | 0.9199 / 0.9429 | 0.1248 / 0.1411 |
| 2 | 2 | zero | 0.9813 / 0.9966 | 0.9958 / 1.0000 | 0.9310 / 0.9424 | 0.9253 / 0.9512 | 0.1236 / 0.1445 |
| 2 | 2 | resample | 0.9104 / 0.9253 | 0.9695 / 0.9819 | 0.7869 / 0.8032 | 0.6427 / 0.6577 | 0.0001 / 0.0000 |
| 3 | 1 | mean | 0.7180 / 0.7368 | 0.9992 / 1.0000 | 0.9993 / 1.0000 | 0.9991 / 1.0000 | 0.2015 / 0.2061 |
| 3 | 1 | zero | 0.6199 / 0.6343 | 0.9992 / 1.0000 | 0.9993 / 1.0000 | 0.9980 / 0.9990 | 0.2292 / 0.2305 |
| 3 | 1 | resample | 0.9985 / 0.9995 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 |
| 3 | 2 | mean | 0.9826 / 0.9907 | 0.9539 / 0.9688 | 0.9572 / 0.9902 | 0.9570 / 0.9614 | 0.1124 / 0.1240 |
| 3 | 2 | zero | 0.9801 / 0.9888 | 0.9419 / 0.9492 | 0.9589 / 0.9888 | 0.9510 / 0.9565 | 0.1156 / 0.1191 |
| 3 | 2 | resample | 0.9166 / 0.9341 | 0.8669 / 0.8784 | 0.7581 / 0.7803 | 0.8795 / 0.8799 | 0.0001 / 0.0000 |
| 4 | 1 | mean | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.3203 / 0.3301 | 0.9993 / 1.0000 | 0.1832 / 0.1812 |
| 4 | 1 | zero | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.2777 / 0.2832 | 0.9993 / 1.0000 | 0.1911 / 0.1909 |
| 4 | 1 | resample | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 |
| 4 | 2 | mean | 0.9736 / 0.9849 | 0.9674 / 0.9937 | 0.9383 / 0.9624 | 0.9835 / 0.9961 | 0.1199 / 0.1260 |
| 4 | 2 | zero | 0.9822 / 0.9941 | 0.9672 / 0.9937 | 0.9386 / 0.9619 | 0.9843 / 0.9966 | 0.1203 / 0.1489 |
| 4 | 2 | resample | 0.9185 / 0.9307 | 0.8335 / 0.8594 | 0.7792 / 0.7969 | 0.9180 / 0.9307 | 0.0001 / 0.0000 |
| 5 | 1 | mean | 0.9994 / 1.0000 | 0.7896 / 0.8047 | 0.9905 / 0.9937 | 0.9993 / 1.0000 | 0.1668 / 0.1680 |
| 5 | 1 | zero | 0.9994 / 1.0000 | 0.7585 / 0.7725 | 0.9846 / 0.9912 | 0.9990 / 1.0000 | 0.1823 / 0.1870 |
| 5 | 1 | resample | 0.9994 / 1.0000 | 0.9960 / 0.9980 | 0.9973 / 0.9985 | 0.9994 / 1.0000 | 0.9994 / 1.0000 |
| 5 | 2 | mean | 0.9260 / 0.9443 | 0.9318 / 0.9678 | 0.9868 / 0.9980 | 0.9517 / 0.9800 | 0.0962 / 0.1094 |
| 5 | 2 | zero | 0.9293 / 0.9453 | 0.9300 / 0.9624 | 0.9861 / 0.9990 | 0.9606 / 0.9868 | 0.0932 / 0.1016 |
| 5 | 2 | resample | 0.8503 / 0.8677 | 0.6239 / 0.6665 | 0.9175 / 0.9438 | 0.8808 / 0.9009 | 0.0001 / 0.0000 |

## Matched donors and other controls

Resampled donors are valid dictionaries matched on query key and query-pair position. Active donors change the queried value; matched-control donors preserve the queried association. An active-versus-matched difference therefore bears on association-specific information, while the other dictionary entries can still differ. Resampling is not interchangeable with zero/mean ablation, and a weak active-donor effect does not prove that the patched heads carry no useful information.

| Seed | Layer | Active donor, all heads P / accuracy | Matched donor, all heads P / accuracy | Matched singleton probability range |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9981–0.9993 |
| 0 | 2 | 0.0001 / 0.0000 | 0.9993 / 1.0000 | 0.9993–0.9993 |
| 1 | 1 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9979–0.9993 |
| 1 | 2 | 0.0001 / 0.0000 | 0.9993 / 1.0000 | 0.9993–0.9993 |
| 2 | 1 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9974–0.9994 |
| 2 | 2 | 0.0001 / 0.0000 | 0.9994 / 1.0000 | 0.9994–0.9994 |
| 3 | 1 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9985–0.9993 |
| 3 | 2 | 0.0001 / 0.0000 | 0.9994 / 1.0000 | 0.9993–0.9993 |
| 4 | 1 | 0.9993 / 1.0000 | 0.9994 / 1.0000 | 0.9993–0.9993 |
| 4 | 2 | 0.0001 / 0.0000 | 0.9994 / 1.0000 | 0.9994–0.9994 |
| 5 | 1 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9960–0.9994 |
| 5 | 2 | 0.0001 / 0.0000 | 0.9994 / 1.0000 | 0.9994–0.9994 |

| Seed | Layer | Wrong-position mean P / accuracy | Wrong-position zero | Wrong-position resample | MLP zero | MLP mean |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.9993 / 1.0000 | 0.9992 / 1.0000 | 0.9993 / 1.0000 | 0.9674 / 0.9736 | 0.9922 / 0.9956 |
| 0 | 2 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9922 / 1.0000 | 0.9935 / 1.0000 |
| 1 | 1 | 0.9994 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9157 / 0.9214 | 0.9881 / 0.9912 |
| 1 | 2 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9931 / 1.0000 | 0.9936 / 1.0000 |
| 2 | 1 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9993 / 1.0000 | 0.9882 / 0.9907 | 0.9992 / 1.0000 |
| 2 | 2 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9911 / 1.0000 | 0.9944 / 1.0000 |
| 3 | 1 | 0.9994 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9745 / 0.9785 | 0.9930 / 0.9946 |
| 3 | 2 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9942 / 1.0000 | 0.9939 / 1.0000 |
| 4 | 1 | 0.9994 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9481 / 0.9551 | 0.9800 / 0.9849 |
| 4 | 2 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9993 / 1.0000 | 0.9900 / 1.0000 | 0.9922 / 1.0000 |
| 5 | 1 | 0.9995 / 1.0000 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9786 / 0.9829 | 0.9986 / 1.0000 |
| 5 | 2 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9994 / 1.0000 | 0.9899 / 1.0000 | 0.9926 / 1.0000 |

Wrong-position controls patch Layer 1 at the query instead of values and Layer 2 at values instead of the query. MLP controls alter a different component at the intended positions; an MLP effect is possible and is not automatically a failed negative control. Full per-condition forecasts, observations, accuracy, and errors are in [forecasts_vs_results.csv](forecasts_vs_results.csv).

## Uncertainty and interpretation

The AI and case-logit-additive forecasts are identical on all 32 cross-layer conditions by construction. Those conditions test their shared cross-layer assumption, not incremental value from the AI explanation. Any advantage over that baseline must come from the within-layer interaction allocation.

Primary loss and loss-difference intervals use 2,000 paired case bootstraps with fixed seed 6200001, preserving correlation across conditions and using shared case weights across model seeds. They are conditional on the six trained models, selected task, calibration means, discovery-derived forecasts, and intervention definitions. They do not incorporate a population distribution over training seeds, uncertainty from model selection, or multiple-task generalization. Displayed observed-condition mean intervals are normal standard-error intervals, not bootstrap intervals; accuracy intervals in the source summary are Wilson intervals. The ±0.10 forecast tolerance is a practical error criterion, not a claimed coverage probability.

Experiment 1's association-lookup account need not predict how information is distributed redundantly across attention heads. Experiment 2 tests that stronger quantitative prediction. A failure here narrows the supported claim even when the task algorithm is correctly identified. Conversely, passing these tests would support this frozen forecast rule on this local task, not a general claim that AI understands arbitrary AI systems. Mean/zero ablations change internal states, and clean-head rescue is a controlled sufficiency test within an altered network. Neither alone establishes strict necessity or sufficiency in ordinary unmodified execution.

All experiment computation used local CPU code; no model inference API was used. Experiment 2 code and data are namespaced under `experiment2/` and `outputs/experiment2/`; frozen Experiment 1 files were not edited.

## Reproduction

Inspect the public registration commit and its manifest before executing. The registered phase controller refuses to overwrite an existing confirmation directory and verifies frozen inputs against the public lock. Archive the existing output directory before a deliberate rerun; use a fresh local working copy or preserve the original tracked outputs. Do not rerun discovery or regenerate calibration means as part of reproducing confirmation.

```sh
# Pure reporting from existing saved results:
.venv/bin/python -m experiment2.report

# Deliberate confirmatory rerun; preserve the originals first:
cp -R outputs/experiment2/confirmatory outputs/experiment2/confirmatory_original
mv outputs/experiment2/confirmatory outputs/experiment2/confirmatory_preserved
.venv/bin/python -m experiment2.run_phase --phase confirmatory
.venv/bin/python -m experiment2.score
.venv/bin/python -m experiment2.report
```

The archive destination names above must not already exist. Reruns create new runtime timestamps; they do not create a new preregistration or replace the original evidentiary run.
