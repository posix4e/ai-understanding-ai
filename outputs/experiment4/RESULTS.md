# Experiment 4: restoring causal key visibility

The full prespecified six-model claim passed.

The correct guard met every registered recovery gate in all six models. The registered advantage over the mean of all eight matched controls passed in all six models.

Validity, recovery and specificity are separate decisions. A positive recovery result does not override a specificity failure, and algebraically guaranteed oracle checks do not count as empirical discoveries.

| Global decision | Passed |
| --- | --- |
| all_six_valid | yes |
| all_six_recovery | yes |
| all_six_specificity | yes |
| full_claim_pass | yes |

## Six-seed result

| Seed | Native accuracy | Grouped accuracy | Guard accuracy | Minimum query accuracy | Guard−grouped P [95% CI] | Guard−mean 8 P [95% CI] | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0000 | 0.5332 | 1.0000 | 1.0000 | 0.4804 [0.4648, 0.4964] | 0.1418 [0.1336, 0.1503] | none |
| 1 | 1.0000 | 0.5645 | 1.0000 | 1.0000 | 0.4412 [0.4280, 0.4547] | 0.1079 [0.1024, 0.1138] | none |
| 2 | 1.0000 | 0.2891 | 1.0000 | 1.0000 | 0.7140 [0.6975, 0.7288] | 0.3463 [0.3376, 0.3543] | none |
| 3 | 1.0000 | 0.3853 | 1.0000 | 1.0000 | 0.6128 [0.5998, 0.6252] | 0.1637 [0.1563, 0.1710] | none |
| 4 | 1.0000 | 0.3408 | 1.0000 | 1.0000 | 0.6680 [0.6512, 0.6850] | 0.2613 [0.2522, 0.2704] | none |
| 5 | 1.0000 | 0.5820 | 0.9995 | 0.9980 | 0.4284 [0.4143, 0.4438] | 0.1058 [0.0984, 0.1133] | none |

Recovery requires accuracy ≥0.95 in every query stratum and a paired probability-improvement lower 95% bound strictly above 0.30. Specificity requires the lower bound versus the equally weighted eight-control mean strictly above 0.05. The full claim requires these plus every validity check in every seed.

## Prospective record and scope

Public preregistration: [b11245db8da95c4a98f9b25a452e828b7c3abe72](https://github.com/posix4e/ai-understanding-ai/commit/b11245db8da95c4a98f9b25a452e828b7c3abe72). Remote contents verified at `2026-10-03T14:45:51.004522+00:00`; confirmation started at `2026-10-03T14:45:59.874058+00:00`. Report generated at `2026-10-03T14:46:10.919908+00:00`.

Experiment 3's failed strong position-label claim is preserved. Its published outcomes motivated this new prospective follow-up; no E4 trained-model discovery or new fitting occurred. Six existing local CPU checkpoints (70,720 parameters, two layers, four heads each) evaluated the same 2,048 fresh dictionaries, balanced at 512 queries per pair index. The data audit excludes prior association dictionaries across all 24 pair orders and four query choices, and excludes repeated unordered dictionaries within E4. All 15 conditions and all failures are retained. No inference API or new training was used.

## Intervention and controls

Grouping all four keys before all four values exposes each first-layer value node to keys it could not see in the training order. Canonical position IDs remain attached to their original tokens. The correct guard removes Vi→Kj edges for j>i, using a stable masked softmax from the grouped model's own cached Q/K. It edits every head's value rows only, uses no native activations or target-dependent decisions, and retains each value's own key.

Nine masks exhaust the row-degree-matched family: V0 keeps only K0; V1 keeps K1 plus A∈{0,2,3}; V2 keeps all keys except B∈{0,1,3}; V3 keeps all keys. Correct is A=0,B=3. Every mask removes six key edges per head with the same per-row counts (3,2,1,0). The masks are not matched on removed attention mass or activation-change norm. V0 has the same mask in every condition, so this comparison cannot isolate its edge identities.

The guard restores first-layer value states by algebra, and the final-query first-layer state is permutation-invariant. Grouped key states remain altered because they lost access to earlier values. Those changed key states can affect layer 2, so final behavioral recovery by the guard alone is empirical. Native key/value-state transplants are explicitly labeled oracle controls. The complete guard-plus-key oracle must reproduce native output; the value-only oracle must reproduce the guard. Neither identity is evidence of a new learned repair mechanism.

## Every matched guard

Accuracy intervals below are individual Wilson 95% intervals, not simultaneous intervals. Probability intervals use shared paired bootstrap rows. Query accuracies are ordered q=0,1,2,3.

| Seed | Guard | Original probability [95% CI] | Accuracy [95% Wilson CI] | Four query accuracies |
| --- | --- | --- | --- | --- |
| 0 | guard_a0_b0 | 0.9280 [0.9183, 0.9368] | 0.9292 [0.9173, 0.9395] | 1.0000, 1.0000, 0.7266, 0.9902 |
| 0 | guard_a0_b1 | 0.9280 [0.9182, 0.9369] | 0.9302 [0.9183, 0.9404] | 1.0000, 1.0000, 0.7305, 0.9902 |
| 0 | guard_a0_b3 | 0.9993 [0.9992, 0.9993] | 1.0000 [0.9981, 1.0000] | 1.0000, 1.0000, 1.0000, 1.0000 |
| 0 | guard_a2_b0 | 0.8692 [0.8562, 0.8814] | 0.8730 [0.8579, 0.8868] | 1.0000, 0.8496, 0.6523, 0.9902 |
| 0 | guard_a2_b1 | 0.8686 [0.8557, 0.8809] | 0.8721 [0.8569, 0.8858] | 1.0000, 0.8496, 0.6484, 0.9902 |
| 0 | guard_a2_b3 | 0.9613 [0.9539, 0.9685] | 0.9639 [0.9549, 0.9711] | 1.0000, 0.8555, 1.0000, 1.0000 |
| 0 | guard_a3_b0 | 0.7514 [0.7371, 0.7654] | 0.7559 [0.7368, 0.7740] | 1.0000, 0.3535, 0.8203, 0.8496 |
| 0 | guard_a3_b1 | 0.7529 [0.7385, 0.7668] | 0.7583 [0.7393, 0.7763] | 1.0000, 0.3633, 0.8184, 0.8516 |
| 0 | guard_a3_b3 | 0.8005 [0.7889, 0.8121] | 0.8052 [0.7875, 0.8218] | 1.0000, 0.3633, 1.0000, 0.8574 |
| 1 | guard_a0_b0 | 0.9990 [0.9983, 0.9993] | 0.9995 [0.9972, 0.9999] | 1.0000, 1.0000, 0.9980, 1.0000 |
| 1 | guard_a0_b1 | 0.9990 [0.9984, 0.9993] | 0.9995 [0.9972, 0.9999] | 1.0000, 1.0000, 0.9980, 1.0000 |
| 1 | guard_a0_b3 | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] | 1.0000, 1.0000, 1.0000, 1.0000 |
| 1 | guard_a2_b0 | 0.8183 [0.8059, 0.8299] | 0.8193 [0.8021, 0.8354] | 1.0000, 0.4160, 0.8613, 1.0000 |
| 1 | guard_a2_b1 | 0.8180 [0.8056, 0.8299] | 0.8198 [0.8026, 0.8359] | 1.0000, 0.4238, 0.8555, 1.0000 |
| 1 | guard_a2_b3 | 0.8288 [0.8172, 0.8403] | 0.8340 [0.8172, 0.8495] | 1.0000, 0.4160, 0.9199, 1.0000 |
| 1 | guard_a3_b0 | 0.8885 [0.8788, 0.8990] | 0.8926 [0.8784, 0.9053] | 1.0000, 0.5938, 0.9980, 0.9785 |
| 1 | guard_a3_b1 | 0.8904 [0.8807, 0.9008] | 0.8955 [0.8815, 0.9080] | 1.0000, 0.6055, 0.9980, 0.9785 |
| 1 | guard_a3_b3 | 0.8890 [0.8792, 0.8994] | 0.8940 [0.8800, 0.9066] | 1.0000, 0.5977, 1.0000, 0.9785 |
| 2 | guard_a0_b0 | 0.8039 [0.7935, 0.8158] | 0.8105 [0.7930, 0.8269] | 1.0000, 1.0000, 0.3613, 0.8809 |
| 2 | guard_a0_b1 | 0.8045 [0.7942, 0.8163] | 0.8110 [0.7935, 0.8274] | 1.0000, 1.0000, 0.3652, 0.8789 |
| 2 | guard_a0_b3 | 0.9994 [0.9994, 0.9994] | 1.0000 [0.9981, 1.0000] | 1.0000, 1.0000, 1.0000, 1.0000 |
| 2 | guard_a2_b0 | 0.5647 [0.5531, 0.5772] | 0.5723 [0.5507, 0.5935] | 1.0000, 0.3672, 0.0410, 0.8809 |
| 2 | guard_a2_b1 | 0.5652 [0.5536, 0.5775] | 0.5723 [0.5507, 0.5935] | 1.0000, 0.3691, 0.0410, 0.8789 |
| 2 | guard_a2_b3 | 0.7558 [0.7431, 0.7689] | 0.7617 [0.7428, 0.7797] | 1.0000, 0.3223, 0.7246, 1.0000 |
| 2 | guard_a3_b0 | 0.5244 [0.5095, 0.5393] | 0.5317 [0.5101, 0.5533] | 1.0000, 0.2402, 0.4004, 0.4863 |
| 2 | guard_a3_b1 | 0.5253 [0.5106, 0.5401] | 0.5332 [0.5116, 0.5547] | 1.0000, 0.2422, 0.4043, 0.4863 |
| 2 | guard_a3_b3 | 0.6802 [0.6678, 0.6916] | 0.6836 [0.6631, 0.7034] | 1.0000, 0.2012, 1.0000, 0.5332 |
| 3 | guard_a0_b0 | 0.9166 [0.9070, 0.9262] | 0.9199 [0.9074, 0.9309] | 1.0000, 1.0000, 0.6973, 0.9824 |
| 3 | guard_a0_b1 | 0.9177 [0.9082, 0.9272] | 0.9209 [0.9084, 0.9318] | 1.0000, 1.0000, 0.7031, 0.9805 |
| 3 | guard_a0_b3 | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] | 1.0000, 1.0000, 1.0000, 1.0000 |
| 3 | guard_a2_b0 | 0.6327 [0.6198, 0.6455] | 0.6353 [0.6142, 0.6558] | 1.0000, 0.3887, 0.1699, 0.9824 |
| 3 | guard_a2_b1 | 0.6348 [0.6222, 0.6478] | 0.6392 [0.6181, 0.6597] | 1.0000, 0.3926, 0.1836, 0.9805 |
| 3 | guard_a2_b3 | 0.7962 [0.7843, 0.8077] | 0.8042 [0.7865, 0.8208] | 1.0000, 0.3652, 0.8516, 1.0000 |
| 3 | guard_a3_b0 | 0.9043 [0.8926, 0.9151] | 0.9097 [0.8965, 0.9213] | 1.0000, 0.9238, 0.7344, 0.9805 |
| 3 | guard_a3_b1 | 0.9054 [0.8940, 0.9162] | 0.9102 [0.8970, 0.9218] | 1.0000, 0.9238, 0.7383, 0.9785 |
| 3 | guard_a3_b3 | 0.9770 [0.9711, 0.9824] | 0.9780 [0.9707, 0.9835] | 1.0000, 0.9141, 1.0000, 0.9980 |
| 4 | guard_a0_b0 | 0.8320 [0.8212, 0.8433] | 0.8408 [0.8243, 0.8560] | 1.0000, 1.0000, 0.4297, 0.9336 |
| 4 | guard_a0_b1 | 0.8334 [0.8229, 0.8448] | 0.8438 [0.8274, 0.8588] | 1.0000, 1.0000, 0.4375, 0.9375 |
| 4 | guard_a0_b3 | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] | 1.0000, 1.0000, 1.0000, 1.0000 |
| 4 | guard_a2_b0 | 0.7244 [0.7105, 0.7383] | 0.7344 [0.7148, 0.7531] | 0.9961, 0.7285, 0.2793, 0.9336 |
| 4 | guard_a2_b1 | 0.7274 [0.7137, 0.7412] | 0.7373 [0.7178, 0.7559] | 0.9961, 0.7285, 0.2871, 0.9375 |
| 4 | guard_a2_b3 | 0.9229 [0.9135, 0.9324] | 0.9253 [0.9131, 0.9359] | 0.9961, 0.7246, 0.9824, 0.9980 |
| 4 | guard_a3_b0 | 0.5734 [0.5579, 0.5888] | 0.5771 [0.5556, 0.5984] | 1.0000, 0.2129, 0.4883, 0.6074 |
| 4 | guard_a3_b1 | 0.5746 [0.5591, 0.5899] | 0.5771 [0.5556, 0.5984] | 1.0000, 0.2168, 0.4863, 0.6055 |
| 4 | guard_a3_b3 | 0.7162 [0.7054, 0.7275] | 0.7192 [0.6994, 0.7383] | 1.0000, 0.2051, 1.0000, 0.6719 |
| 5 | guard_a0_b0 | 0.9440 [0.9353, 0.9521] | 0.9468 [0.9362, 0.9557] | 0.9961, 1.0000, 0.7988, 0.9922 |
| 5 | guard_a0_b1 | 0.9443 [0.9356, 0.9524] | 0.9463 [0.9357, 0.9552] | 0.9961, 1.0000, 0.7969, 0.9922 |
| 5 | guard_a0_b3 | 0.9985 [0.9972, 0.9994] | 0.9995 [0.9972, 0.9999] | 0.9980, 1.0000, 1.0000, 1.0000 |
| 5 | guard_a2_b0 | 0.9229 [0.9128, 0.9328] | 0.9268 [0.9147, 0.9373] | 0.9961, 0.9609, 0.7578, 0.9922 |
| 5 | guard_a2_b1 | 0.9231 [0.9131, 0.9331] | 0.9268 [0.9147, 0.9373] | 0.9961, 0.9629, 0.7559, 0.9922 |
| 5 | guard_a2_b3 | 0.9898 [0.9859, 0.9932] | 0.9922 [0.9873, 0.9952] | 0.9980, 0.9707, 1.0000, 1.0000 |
| 5 | guard_a3_b0 | 0.7900 [0.7763, 0.8029] | 0.8008 [0.7829, 0.8175] | 0.9941, 0.4844, 0.8457, 0.8789 |
| 5 | guard_a3_b1 | 0.7916 [0.7780, 0.8047] | 0.8027 [0.7849, 0.8194] | 0.9941, 0.4902, 0.8457, 0.8809 |
| 5 | guard_a3_b3 | 0.8359 [0.8245, 0.8474] | 0.8462 [0.8299, 0.8612] | 0.9980, 0.4941, 1.0000, 0.8926 |

Every condition, including references and oracles, is in [condition_results.csv](condition_results.csv). Complete query-stratified means, probability intervals and Wilson intervals are in [per_query_results.csv](per_query_results.csv). No low-performing guard or query stratum was dropped.

### Mean and best matched controls

| Seed | Mean of 8 wrong masks P [95% CI] | Best point mask | Best point P | Guard−best P [reselected 95% CI] |
| --- | --- | --- | --- | --- |
| 0 | 0.8575 [0.8490, 0.8656] | guard_a2_b3 | 0.9613 | 0.0379 [0.0307, 0.0454] |
| 1 | 0.8914 [0.8855, 0.8969] | guard_a0_b1 | 0.9990 | 0.0003 [0.0000, 0.0008] |
| 2 | 0.6530 [0.6451, 0.6618] | guard_a0_b1 | 0.8045 | 0.1949 [0.1830, 0.2052] |
| 3 | 0.8356 [0.8283, 0.8431] | guard_a3_b3 | 0.9770 | 0.0224 [0.0169, 0.0282] |
| 4 | 0.7380 [0.7289, 0.7471] | guard_a2_b3 | 0.9229 | 0.0765 [0.0669, 0.0859] |
| 5 | 0.8927 [0.8853, 0.9004] | guard_a2_b3 | 0.9898 | 0.0087 [0.0054, 0.0124] |

The best-control interval reselects the highest-mean wrong mask within each shared bootstrap draw. It is descriptive and is not an extra success gate. Passing the mean-control comparison does not establish that the correct guard beats every individual mask or is uniquely effective. All eight paired guard-minus-control effects and intervals are in [control_comparisons.csv](control_comparisons.csv).

## Validity and oracle checks

| Seed | Native accuracy | Grouped no-op max P error | Guard L1 value max error | Guard L1 query max error | Guard key change vs grouped | Guard−value oracle max class-P error | Full oracle−native max class-P error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0000 | 0 | 1.19e-06 | 1.91e-06 | 0 | 1.19e-07 | 1.19e-07 |
| 1 | 1.0000 | 0 | 1.43e-06 | 1.43e-06 | 0 | 1.19e-07 | 1.19e-07 |
| 2 | 1.0000 | 0 | 1.43e-06 | 9.54e-07 | 0 | 1.19e-07 | 1.19e-07 |
| 3 | 1.0000 | 0 | 1.43e-06 | 9.54e-07 | 0 | 1.19e-07 | 1.19e-07 |
| 4 | 1.0000 | 0 | 1.67e-06 | 9.54e-07 | 0 | 1.19e-07 | 1.19e-07 |
| 5 | 1.0000 | 0 | 1.43e-06 | 9.54e-07 | 0 | 2.98e-07 | 1.19e-07 |

Native accuracy must reach 0.95. Grouped no-op error must be ≤10⁻⁶; the five residual/output identities must be ≤10⁻⁵. Output-oracle errors cover every case and all 16 classes, rather than only the correct class. Finite complete arrays and full-probability/argmax/accuracy consistency are also checked. Actual first-layer output diagnostics and effective layer-2 input diagnostics are stored separately: transplanting a state at layer-2 input does not retroactively change the first-layer output.

## Keys and guard factorial (descriptive)

| Seed | Keys effect without guard [95% CI] | Keys effect with guard [95% CI] | Interaction: full−guard−keys+grouped [95% CI] |
| --- | --- | --- | --- |
| 0 | 0.0113 [0.0060, 0.0161] | 0.0000 [0.0000, 0.0000] | -0.0113 [-0.0161, -0.0060] |
| 1 | -0.0012 [-0.0048, 0.0022] | 0.0000 [0.0000, 0.0001] | 0.0013 [-0.0022, 0.0048] |
| 2 | 0.0011 [-0.0023, 0.0044] | 0.0000 [0.0000, 0.0000] | -0.0011 [-0.0044, 0.0023] |
| 3 | 0.0018 [0.0002, 0.0033] | 0.0000 [0.0000, 0.0000] | -0.0018 [-0.0033, -0.0002] |
| 4 | 0.0088 [0.0043, 0.0135] | 0.0000 [0.0000, 0.0000] | -0.0088 [-0.0135, -0.0043] |
| 5 | 0.0124 [0.0080, 0.0173] | 0.0009 [0.0001, 0.0022] | -0.0115 [-0.0166, -0.0067] |

These paired probability-scale effects compare grouped, guard, native-keys restoration, and combined guard-plus-native-keys restoration. They help describe remaining dependencies and carry no new gate. The combined condition is an oracle with an algebraic output-restoration guarantee.

## All 180 prospective numeric forecasts

The frozen predictor assigned original-target probability and accuracy to 15 conditions in each of six models. Every error is evaluated against absolute tolerance 0.15, separately from causal gates. Reference and oracle forecasts are explicitly separated so known identities do not inflate claims about predicting novel intervention effects.

| Scope | Forecast count | Within ±0.15 | Fraction | MAE | RMSE | Maximum error |
| --- | --- | --- | --- | --- | --- | --- |
| all_15 | 180 | 176 | 97.8% | 0.0441 | 0.0672 | 0.2277 |
| matched_guards_9 | 108 | 104 | 96.3% | 0.0690 | 0.0863 | 0.2277 |
| oracles_3 | 36 | 36 | 100.0% | 0.0087 | 0.0116 | 0.0249 |
| references_3 | 36 | 36 | 100.0% | 0.0047 | 0.0082 | 0.0239 |

| Seed | All forecasts within ±0.15 | Guard forecasts within ±0.15 | All RMSE | Guard RMSE |
| --- | --- | --- | --- | --- |
| 0 | 30/30 | 18/18 | 0.0553 | 0.0711 |
| 1 | 30/30 | 18/18 | 0.0618 | 0.0787 |
| 2 | 30/30 | 18/18 | 0.0578 | 0.0744 |
| 3 | 26/30 | 14/18 | 0.0975 | 0.1257 |
| 4 | 30/30 | 18/18 | 0.0559 | 0.0717 |
| 5 | 30/30 | 18/18 | 0.0649 | 0.0835 |

Every forecast is in [forecasts_vs_results.csv](forecasts_vs_results.csv); every miss is listed in [forecast_misses.csv](forecast_misses.csv). This error tolerance is a practical criterion, not a forecast-coverage probability.

## Attention and uncertainty

All four attention heads remain visible without selecting favorable ones. Per-condition head means for first-layer value→true-key attention and second-layer query→correct-value attention are in [attention_diagnostics.csv](attention_diagnostics.csv). They are descriptive measurements, not additional causal tests. Residual diagnostics, every query contrast, and all gate decisions remain in [scores.json](confirmatory/scores.json).

Mean intervals use 2,000 dictionary-row multinomial bootstraps, stratified within the four query indices with fixed seed 8200001. The same weights are shared across every condition and model. The eight wrong controls are averaged within each row before resampling. Uncertainty is conditional on these six checkpoints and input distribution, not a population distribution over trained models. Individual Wilson and condition intervals are not simultaneous intervals.

## Interpretation and preservation

The stated claim concerns empirical sufficiency of a specific causal-visibility repair and its advantage over the mean of the complete row-degree-matched control family. It does not establish necessity of every removed edge, a unique complete algorithm, spontaneous grouped-layout transfer, or a universal transformer mechanism. The guard supplies an externally chosen training-prefix mask, so it is not a learned adaptation by the model.

This is a prospective extension of positional-binding work, with no field-wide novelty claim; see [the earlier primary-source audit](../experiment3/RELATED_WORK.md). All E1–E3 frozen results, including E3's negative primary result, remain unchanged. Positive subresults here do not relabel earlier failures. Any subsequent refinement requires its own prospective data and registration.

## Reproduction

The registration fixes inputs, checkpoint hashes, intervention code, forecasts and score rules. These commands only recompute scores and reporting from saved outcomes; preserve existing generated files before a deliberate rerun if their original runtime record is needed. The trained-model phase controller separately verifies the public lock and refuses an existing confirmation directory.

```sh
.venv/bin/python -m experiment4.score
.venv/bin/python -m experiment4.report
```
