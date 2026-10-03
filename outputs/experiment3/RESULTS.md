# Experiment 3: positional coordinates and key–value binding

The full prespecified six-model mechanistic claim did not pass.

| Global decision | Passed |
| --- | --- |
| all_six_valid | yes |
| all_six_mechanism_gates | no |
| full_claim_pass | no |

Numeric forecast accuracy is a separate result from the registered mechanistic gates; all forecast misses and every failed condition are retained below and in the accompanying files.

This study is a prospective test and systematic extension of prior positional-steering work. It makes no priority claim for discovering positional key–value binding; the distinctive evidence sought here is the frozen permutation grid, specifically predicted wrong answers, matched coherent controls, and replication across six existing models.

The [primary-source related-work audit](RELATED_WORK.md) identifies close precedents and explains the narrower scope of this test.

Preregistration: [42fd4236ecfa0535dacf9ddf6c824c41bd66f02e](https://github.com/posix4e/ai-understanding-ai/commit/42fd4236ecfa0535dacf9ddf6c824c41bd66f02e). Public contents verified at `2026-10-03T14:25:09.329884+00:00`; confirmation started at `2026-10-03T14:28:14.516699+00:00`. Report generated at `2026-10-03T14:28:55.214665+00:00`.

## What was tested

The same six frozen, locally trained CPU transformers from Experiments 1–2 were reused; each has 70,720 parameters, two layers and four attention heads per layer. No E3 training, shifted-layout discovery, or per-model fitting occurred. Each of 2,048 fresh dictionaries contributes one query, with 512 cases at each of four query-pair indices. All 50 conditions use the same rows in every model.

The original sequence alternates four keys and values before a final query. The grouped layout presents all four keys, then all four values, then the query. Canonical positional coordinates travel with their original tokens. For a permutation π, a coherent map assigns key/value pair i coordinates 2π(i) and 2π(i)+1; a value-only map leaves key i at 2i but assigns its value coordinate 2π(i)+1. The binding account predicts the value at inverse-π(query-pair) for value-only maps, and the original answer for coherent maps. All keys precede all values, and query token/position remain unchanged. Input residual patching changes the supplied token-plus-position embedding sum; model weights remain frozen.

All 24 permutations were fixed independently of outcomes. Collapsing identity maps leaves 23 coherent and 23 value-only maps plus four references. The primary test uses all nine value derangements and their nine matched coherent controls. Each value condition must reach slot accuracy ≥0.90, each coherent control original accuracy ≥0.95, and each seed's equally weighted nine-condition mean of slot minus original probability must have a paired 95% bootstrap lower bound strictly above 0.80. Validity also requires native original accuracy ≥0.95 and maximum no-op original-probability error ≤10⁻⁶. Every condition and all six seeds must pass; averages cannot override individual failures.

## Seed-by-seed gates

| Seed | Checkpoint cohort | Native accuracy | Minimum of 9 slot accuracies | Minimum of 9 coherent accuracies | Slot−original P [95% CI] | No-op max error | Failed gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | E1 | 1.0000 | 0.5347 | 0.4951 | 0.4350 [0.4176, 0.4522] | 0 | every_value_derangement, every_coherent_control, probability_margin_ci95_lower |
| 1 | E1 | 1.0000 | 0.5747 | 0.5640 | 0.4676 [0.4515, 0.4827] | 0 | every_value_derangement, every_coherent_control, probability_margin_ci95_lower |
| 2 | E1 | 1.0000 | 0.2729 | 0.2896 | 0.1069 [0.0885, 0.1252] | 0 | every_value_derangement, every_coherent_control, probability_margin_ci95_lower |
| 3 | E2 replication | 1.0000 | 0.3823 | 0.3799 | 0.2118 [0.1978, 0.2265] | 0 | every_value_derangement, every_coherent_control, probability_margin_ci95_lower |
| 4 | E2 replication | 1.0000 | 0.3276 | 0.3027 | 0.1708 [0.1543, 0.1872] | 0 | every_value_derangement, every_coherent_control, probability_margin_ci95_lower |
| 5 | E2 replication | 1.0000 | 0.5806 | 0.5552 | 0.4866 [0.4720, 0.5005] | 0 | every_value_derangement, every_coherent_control, probability_margin_ci95_lower |

### Every primary condition

Accuracy intervals are individual Wilson 95% intervals. They are not simultaneous intervals across conditions or models; the registered accuracy gates use the point estimates.

| Seed | Condition | Endpoint | Accuracy [individual 95% CI] | Minimum | Decision |
| --- | --- | --- | --- | --- | --- |
| 0 | value_1032 | slot_accuracy | 0.5439 [0.5223, 0.5654] | 0.9 | FAIL |
| 0 | value_1230 | slot_accuracy | 0.5430 [0.5213, 0.5644] | 0.9 | FAIL |
| 0 | value_1302 | slot_accuracy | 0.5415 [0.5199, 0.5630] | 0.9 | FAIL |
| 0 | value_2031 | slot_accuracy | 0.5508 [0.5292, 0.5722] | 0.9 | FAIL |
| 0 | value_2301 | slot_accuracy | 0.5532 [0.5316, 0.5746] | 0.9 | FAIL |
| 0 | value_2310 | slot_accuracy | 0.5493 [0.5277, 0.5708] | 0.9 | FAIL |
| 0 | value_3012 | slot_accuracy | 0.5347 [0.5130, 0.5562] | 0.9 | FAIL |
| 0 | value_3201 | slot_accuracy | 0.5483 [0.5267, 0.5698] | 0.9 | FAIL |
| 0 | value_3210 | slot_accuracy | 0.5420 [0.5204, 0.5635] | 0.9 | FAIL |
| 0 | coherent_1032 | original_accuracy | 0.5142 [0.4925, 0.5358] | 0.95 | FAIL |
| 0 | coherent_1230 | original_accuracy | 0.5176 [0.4959, 0.5392] | 0.95 | FAIL |
| 0 | coherent_1302 | original_accuracy | 0.5371 [0.5155, 0.5586] | 0.95 | FAIL |
| 0 | coherent_2031 | original_accuracy | 0.5010 [0.4793, 0.5226] | 0.95 | FAIL |
| 0 | coherent_2301 | original_accuracy | 0.4995 [0.4779, 0.5211] | 0.95 | FAIL |
| 0 | coherent_2310 | original_accuracy | 0.4951 [0.4735, 0.5168] | 0.95 | FAIL |
| 0 | coherent_3012 | original_accuracy | 0.5278 [0.5062, 0.5494] | 0.95 | FAIL |
| 0 | coherent_3201 | original_accuracy | 0.5337 [0.5120, 0.5552] | 0.95 | FAIL |
| 0 | coherent_3210 | original_accuracy | 0.5317 [0.5101, 0.5533] | 0.95 | FAIL |
| 1 | value_1032 | slot_accuracy | 0.5747 [0.5532, 0.5960] | 0.9 | FAIL |
| 1 | value_1230 | slot_accuracy | 0.5830 [0.5615, 0.6042] | 0.9 | FAIL |
| 1 | value_1302 | slot_accuracy | 0.5845 [0.5630, 0.6056] | 0.9 | FAIL |
| 1 | value_2031 | slot_accuracy | 0.5781 [0.5566, 0.5993] | 0.9 | FAIL |
| 1 | value_2301 | slot_accuracy | 0.5840 [0.5625, 0.6052] | 0.9 | FAIL |
| 1 | value_2310 | slot_accuracy | 0.5776 [0.5561, 0.5989] | 0.9 | FAIL |
| 1 | value_3012 | slot_accuracy | 0.5767 [0.5551, 0.5979] | 0.9 | FAIL |
| 1 | value_3201 | slot_accuracy | 0.5874 [0.5659, 0.6085] | 0.9 | FAIL |
| 1 | value_3210 | slot_accuracy | 0.5859 [0.5645, 0.6071] | 0.9 | FAIL |
| 1 | coherent_1032 | original_accuracy | 0.5742 [0.5527, 0.5955] | 0.95 | FAIL |
| 1 | coherent_1230 | original_accuracy | 0.5786 [0.5571, 0.5998] | 0.95 | FAIL |
| 1 | coherent_1302 | original_accuracy | 0.5640 [0.5424, 0.5853] | 0.95 | FAIL |
| 1 | coherent_2031 | original_accuracy | 0.5894 [0.5679, 0.6105] | 0.95 | FAIL |
| 1 | coherent_2301 | original_accuracy | 0.5825 [0.5610, 0.6037] | 0.95 | FAIL |
| 1 | coherent_2310 | original_accuracy | 0.5693 [0.5478, 0.5906] | 0.95 | FAIL |
| 1 | coherent_3012 | original_accuracy | 0.5762 [0.5546, 0.5974] | 0.95 | FAIL |
| 1 | coherent_3201 | original_accuracy | 0.5669 [0.5453, 0.5882] | 0.95 | FAIL |
| 1 | coherent_3210 | original_accuracy | 0.5918 [0.5704, 0.6129] | 0.95 | FAIL |
| 2 | value_1032 | slot_accuracy | 0.2729 [0.2541, 0.2927] | 0.9 | FAIL |
| 2 | value_1230 | slot_accuracy | 0.2749 [0.2560, 0.2946] | 0.9 | FAIL |
| 2 | value_1302 | slot_accuracy | 0.2749 [0.2560, 0.2946] | 0.9 | FAIL |
| 2 | value_2031 | slot_accuracy | 0.2812 [0.2622, 0.3011] | 0.9 | FAIL |
| 2 | value_2301 | slot_accuracy | 0.2847 [0.2655, 0.3046] | 0.9 | FAIL |
| 2 | value_2310 | slot_accuracy | 0.2920 [0.2727, 0.3121] | 0.9 | FAIL |
| 2 | value_3012 | slot_accuracy | 0.2866 [0.2675, 0.3066] | 0.9 | FAIL |
| 2 | value_3201 | slot_accuracy | 0.2886 [0.2694, 0.3086] | 0.9 | FAIL |
| 2 | value_3210 | slot_accuracy | 0.2832 [0.2641, 0.3031] | 0.9 | FAIL |
| 2 | coherent_1032 | original_accuracy | 0.3047 [0.2851, 0.3250] | 0.95 | FAIL |
| 2 | coherent_1230 | original_accuracy | 0.3110 [0.2914, 0.3314] | 0.95 | FAIL |
| 2 | coherent_1302 | original_accuracy | 0.3350 [0.3148, 0.3557] | 0.95 | FAIL |
| 2 | coherent_2031 | original_accuracy | 0.2896 [0.2703, 0.3096] | 0.95 | FAIL |
| 2 | coherent_2301 | original_accuracy | 0.3066 [0.2870, 0.3270] | 0.95 | FAIL |
| 2 | coherent_2310 | original_accuracy | 0.2979 [0.2784, 0.3180] | 0.95 | FAIL |
| 2 | coherent_3012 | original_accuracy | 0.3398 [0.3196, 0.3606] | 0.95 | FAIL |
| 2 | coherent_3201 | original_accuracy | 0.3394 [0.3192, 0.3601] | 0.95 | FAIL |
| 2 | coherent_3210 | original_accuracy | 0.3306 [0.3105, 0.3512] | 0.95 | FAIL |
| 3 | value_1032 | slot_accuracy | 0.3838 [0.3630, 0.4050] | 0.9 | FAIL |
| 3 | value_1230 | slot_accuracy | 0.3843 [0.3634, 0.4055] | 0.9 | FAIL |
| 3 | value_1302 | slot_accuracy | 0.3857 [0.3649, 0.4070] | 0.9 | FAIL |
| 3 | value_2031 | slot_accuracy | 0.3926 [0.3716, 0.4139] | 0.9 | FAIL |
| 3 | value_2301 | slot_accuracy | 0.3887 [0.3678, 0.4100] | 0.9 | FAIL |
| 3 | value_2310 | slot_accuracy | 0.3848 [0.3639, 0.4060] | 0.9 | FAIL |
| 3 | value_3012 | slot_accuracy | 0.3823 [0.3615, 0.4036] | 0.9 | FAIL |
| 3 | value_3201 | slot_accuracy | 0.3857 [0.3649, 0.4070] | 0.9 | FAIL |
| 3 | value_3210 | slot_accuracy | 0.3823 [0.3615, 0.4036] | 0.9 | FAIL |
| 3 | coherent_1032 | original_accuracy | 0.3799 [0.3591, 0.4011] | 0.95 | FAIL |
| 3 | coherent_1230 | original_accuracy | 0.3945 [0.3736, 0.4159] | 0.95 | FAIL |
| 3 | coherent_1302 | original_accuracy | 0.3823 [0.3615, 0.4036] | 0.95 | FAIL |
| 3 | coherent_2031 | original_accuracy | 0.4097 [0.3886, 0.4311] | 0.95 | FAIL |
| 3 | coherent_2301 | original_accuracy | 0.3936 [0.3726, 0.4149] | 0.95 | FAIL |
| 3 | coherent_2310 | original_accuracy | 0.4102 [0.3890, 0.4316] | 0.95 | FAIL |
| 3 | coherent_3012 | original_accuracy | 0.3887 [0.3678, 0.4100] | 0.95 | FAIL |
| 3 | coherent_3201 | original_accuracy | 0.4009 [0.3799, 0.4223] | 0.95 | FAIL |
| 3 | coherent_3210 | original_accuracy | 0.4155 [0.3944, 0.4370] | 0.95 | FAIL |
| 4 | value_1032 | slot_accuracy | 0.3311 [0.3110, 0.3517] | 0.9 | FAIL |
| 4 | value_1230 | slot_accuracy | 0.3384 [0.3182, 0.3592] | 0.9 | FAIL |
| 4 | value_1302 | slot_accuracy | 0.3364 [0.3163, 0.3572] | 0.9 | FAIL |
| 4 | value_2031 | slot_accuracy | 0.3276 [0.3076, 0.3483] | 0.9 | FAIL |
| 4 | value_2301 | slot_accuracy | 0.3340 [0.3139, 0.3547] | 0.9 | FAIL |
| 4 | value_2310 | slot_accuracy | 0.3413 [0.3211, 0.3621] | 0.9 | FAIL |
| 4 | value_3012 | slot_accuracy | 0.3325 [0.3124, 0.3532] | 0.9 | FAIL |
| 4 | value_3201 | slot_accuracy | 0.3291 [0.3091, 0.3498] | 0.9 | FAIL |
| 4 | value_3210 | slot_accuracy | 0.3306 [0.3105, 0.3512] | 0.9 | FAIL |
| 4 | coherent_1032 | original_accuracy | 0.3335 [0.3134, 0.3542] | 0.95 | FAIL |
| 4 | coherent_1230 | original_accuracy | 0.3301 [0.3100, 0.3507] | 0.95 | FAIL |
| 4 | coherent_1302 | original_accuracy | 0.3408 [0.3206, 0.3616] | 0.95 | FAIL |
| 4 | coherent_2031 | original_accuracy | 0.3027 [0.2832, 0.3230] | 0.95 | FAIL |
| 4 | coherent_2301 | original_accuracy | 0.3247 [0.3048, 0.3453] | 0.95 | FAIL |
| 4 | coherent_2310 | original_accuracy | 0.3105 [0.2909, 0.3309] | 0.95 | FAIL |
| 4 | coherent_3012 | original_accuracy | 0.3433 [0.3230, 0.3641] | 0.95 | FAIL |
| 4 | coherent_3201 | original_accuracy | 0.3550 [0.3345, 0.3760] | 0.95 | FAIL |
| 4 | coherent_3210 | original_accuracy | 0.3267 [0.3067, 0.3473] | 0.95 | FAIL |
| 5 | value_1032 | slot_accuracy | 0.5894 [0.5679, 0.6105] | 0.9 | FAIL |
| 5 | value_1230 | slot_accuracy | 0.5835 [0.5620, 0.6047] | 0.9 | FAIL |
| 5 | value_1302 | slot_accuracy | 0.5864 [0.5650, 0.6076] | 0.9 | FAIL |
| 5 | value_2031 | slot_accuracy | 0.5884 [0.5669, 0.6095] | 0.9 | FAIL |
| 5 | value_2301 | slot_accuracy | 0.5845 [0.5630, 0.6056] | 0.9 | FAIL |
| 5 | value_2310 | slot_accuracy | 0.5806 [0.5591, 0.6018] | 0.9 | FAIL |
| 5 | value_3012 | slot_accuracy | 0.5879 [0.5664, 0.6090] | 0.9 | FAIL |
| 5 | value_3201 | slot_accuracy | 0.5884 [0.5669, 0.6095] | 0.9 | FAIL |
| 5 | value_3210 | slot_accuracy | 0.5830 [0.5615, 0.6042] | 0.9 | FAIL |
| 5 | coherent_1032 | original_accuracy | 0.5669 [0.5453, 0.5882] | 0.95 | FAIL |
| 5 | coherent_1230 | original_accuracy | 0.5552 [0.5336, 0.5766] | 0.95 | FAIL |
| 5 | coherent_1302 | original_accuracy | 0.5747 [0.5532, 0.5960] | 0.95 | FAIL |
| 5 | coherent_2031 | original_accuracy | 0.5825 [0.5610, 0.6037] | 0.95 | FAIL |
| 5 | coherent_2301 | original_accuracy | 0.6099 [0.5886, 0.6308] | 0.95 | FAIL |
| 5 | coherent_2310 | original_accuracy | 0.5981 [0.5767, 0.6192] | 0.95 | FAIL |
| 5 | coherent_3012 | original_accuracy | 0.6011 [0.5797, 0.6221] | 0.95 | FAIL |
| 5 | coherent_3201 | original_accuracy | 0.6055 [0.5841, 0.6264] | 0.95 | FAIL |
| 5 | coherent_3210 | original_accuracy | 0.6128 [0.5915, 0.6337] | 0.95 | FAIL |

### Query-stratified probability margin

| Seed | Query-pair index | Mean of 9 slot−original probabilities [paired 95% CI] |
| --- | --- | --- |
| 0 | 0 | 0.1099 [0.0774, 0.1454] |
| 0 | 1 | 0.2342 [0.1919, 0.2752] |
| 0 | 2 | 0.8357 [0.8078, 0.8626] |
| 0 | 3 | 0.5602 [0.5274, 0.5922] |
| 1 | 0 | 0.1718 [0.1274, 0.2158] |
| 1 | 1 | 0.0029 [-0.0305, 0.0375] |
| 1 | 2 | 0.9570 [0.9460, 0.9672] |
| 1 | 3 | 0.7385 [0.7085, 0.7683] |
| 2 | 0 | 0.1511 [0.1135, 0.1898] |
| 2 | 1 | -0.0611 [-0.0874, -0.0329] |
| 2 | 2 | 0.0406 [0.0050, 0.0741] |
| 2 | 3 | 0.2970 [0.2578, 0.3363] |
| 3 | 0 | -0.0365 [-0.0631, -0.0095] |
| 3 | 1 | 0.0439 [0.0092, 0.0827] |
| 3 | 2 | -0.1397 [-0.1743, -0.1055] |
| 3 | 3 | 0.9795 [0.9720, 0.9863] |
| 4 | 0 | 0.1404 [0.1056, 0.1759] |
| 4 | 1 | 0.0282 [-0.0071, 0.0638] |
| 4 | 2 | 0.1832 [0.1532, 0.2150] |
| 4 | 3 | 0.3314 [0.2981, 0.3638] |
| 5 | 0 | -0.0500 [-0.0768, -0.0218] |
| 5 | 1 | 0.3687 [0.3318, 0.4031] |
| 5 | 2 | 0.7609 [0.7342, 0.7847] |
| 5 | 3 | 0.8667 [0.8467, 0.8868] |

Complete per-query probability intervals and accuracy intervals for every condition are included in [scores.json](confirmatory/scores.json). Query subgroup rows are descriptive; they do not add new success gates.

## Native, canonical and no-op references

| Seed | Reference | Original probability [95% CI] | Original accuracy [95% CI] |
| --- | --- | --- | --- |
| 0 | original_native | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] |
| 0 | original_noop | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] |
| 0 | grouped_native | 0.1941 [0.1808, 0.2086] | 0.2017 [0.1848, 0.2196] |
| 0 | grouped_canonical | 0.5223 [0.5059, 0.5392] | 0.5391 [0.5174, 0.5606] |
| 1 | original_native | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] |
| 1 | original_noop | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] |
| 1 | grouped_native | 0.1847 [0.1715, 0.1986] | 0.1929 [0.1764, 0.2105] |
| 1 | grouped_canonical | 0.5752 [0.5605, 0.5890] | 0.5884 [0.5669, 0.6095] |
| 2 | original_native | 0.9994 [0.9994, 0.9994] | 1.0000 [0.9981, 1.0000] |
| 2 | original_noop | 0.9994 [0.9994, 0.9994] | 1.0000 [0.9981, 1.0000] |
| 2 | grouped_native | 0.2038 [0.1891, 0.2187] | 0.2085 [0.1915, 0.2266] |
| 2 | grouped_canonical | 0.2773 [0.2616, 0.2929] | 0.2886 [0.2694, 0.3086] |
| 3 | original_native | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] |
| 3 | original_noop | 0.9993 [0.9993, 0.9993] | 1.0000 [0.9981, 1.0000] |
| 3 | grouped_native | 0.1930 [0.1784, 0.2069] | 0.1978 [0.1811, 0.2156] |
| 3 | grouped_canonical | 0.3754 [0.3635, 0.3877] | 0.3848 [0.3639, 0.4060] |
| 4 | original_native | 0.9994 [0.9993, 0.9994] | 1.0000 [0.9981, 1.0000] |
| 4 | original_noop | 0.9994 [0.9993, 0.9994] | 1.0000 [0.9981, 1.0000] |
| 4 | grouped_native | 0.1969 [0.1829, 0.2113] | 0.2002 [0.1834, 0.2181] |
| 4 | grouped_canonical | 0.3303 [0.3140, 0.3474] | 0.3315 [0.3115, 0.3522] |
| 5 | original_native | 0.9994 [0.9994, 0.9994] | 1.0000 [0.9981, 1.0000] |
| 5 | original_noop | 0.9994 [0.9994, 0.9994] | 1.0000 [0.9981, 1.0000] |
| 5 | grouped_native | 0.1845 [0.1708, 0.1990] | 0.1929 [0.1764, 0.2105] |
| 5 | grouped_canonical | 0.5722 [0.5566, 0.5875] | 0.5845 [0.5630, 0.6056] |

Grouped-native has no parity-based slot map. Its slot fields duplicate the original reference by convention and are marked `slot_mapping_defined=false`; they are not evidence for the coordinate rule.

## All prospective numeric forecasts

Four means were forecast for each condition and each model: original probability, slot probability, original accuracy, and slot accuracy. The full grid therefore contains 1,200 forecasts. Each uses the fixed absolute-error tolerance 0.15. This is a practical tolerance, not a probabilistic coverage interval. Primary-only summaries distinguish all 18 primary maps from the nine value derangements alone.

| Scope | Forecasts | Within ±0.15 | Fraction | MAE | RMSE | Maximum error |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1200 | 263 | 21.9% | 0.3939 | 0.4485 | 0.6971 |
| primary_18 | 432 | 86 | 19.9% | 0.4107 | 0.4575 | 0.6971 |
| primary_value_9 | 216 | 86 | 39.8% | 0.3116 | 0.3823 | 0.6971 |

| Seed | Scope | Within ±0.15 | RMSE | Maximum error |
| --- | --- | --- | --- | --- |
| 0 | all | 48/200 | 0.3744 | 0.4827 |
| 0 | primary_18 | 18/72 | 0.3840 | 0.4749 |
| 0 | primary_value_9 | 18/36 | 0.3035 | 0.4353 |
| 1 | all | 51/200 | 0.3310 | 0.4109 |
| 1 | primary_18 | 18/72 | 0.3370 | 0.4060 |
| 1 | primary_value_9 | 18/36 | 0.2781 | 0.3953 |
| 2 | all | 38/200 | 0.5667 | 0.6971 |
| 2 | primary_18 | 10/72 | 0.5762 | 0.6971 |
| 2 | primary_value_9 | 10/36 | 0.4944 | 0.6971 |
| 3 | all | 37/200 | 0.4907 | 0.5974 |
| 3 | primary_18 | 9/72 | 0.4996 | 0.5901 |
| 3 | primary_value_9 | 9/36 | 0.4243 | 0.5877 |
| 4 | all | 41/200 | 0.5381 | 0.6844 |
| 4 | primary_18 | 13/72 | 0.5526 | 0.6673 |
| 4 | primary_value_9 | 13/36 | 0.4567 | 0.6424 |
| 5 | all | 48/200 | 0.3251 | 0.4236 |
| 5 | primary_18 | 18/72 | 0.3288 | 0.4148 |
| 5 | primary_value_9 | 18/36 | 0.2732 | 0.3894 |

Every forecast is in [forecasts_vs_results.csv](forecasts_vs_results.csv). Every miss, without filtering by condition family or seed, is in [forecast_misses.csv](forecast_misses.csv). Native references and no-op checks are not novel mechanistic successes.

## Baseline fidelity to observed answers

Fidelity asks whether a rule predicts the model's observed argmax, which differs from task correctness. The semantic rule always chooses the original associated value; the slot rule chooses the coordinate-implied value. A uniform displayed-value rule assigns probability 1/4 when the observed argmax is among the four displayed values, otherwise zero. Uniform over all 16 classes assigns 1/16. The slot rule is the tested mechanistic prediction, not an independent competitor. Native-grouped slot fidelity is undefined as a mechanistic rule despite its reference fields.

| Seed | Rule | Mean fidelity across 9 value derangements |
| --- | --- | --- |
| 0 | semantic_original | 0.1027 |
| 0 | uniform_displayed_four | 0.2090 |
| 0 | uniform_sixteen | 0.0625 |
| 0 | slot_rule | 0.5452 |
| 1 | semantic_original | 0.1086 |
| 1 | uniform_displayed_four | 0.2271 |
| 1 | uniform_sixteen | 0.0625 |
| 1 | slot_rule | 0.5813 |
| 2 | semantic_original | 0.1700 |
| 2 | uniform_displayed_four | 0.1975 |
| 2 | uniform_sixteen | 0.0625 |
| 2 | slot_rule | 0.2821 |
| 3 | semantic_original | 0.1732 |
| 3 | uniform_displayed_four | 0.2245 |
| 3 | uniform_sixteen | 0.0625 |
| 3 | slot_rule | 0.3856 |
| 4 | semantic_original | 0.1466 |
| 4 | uniform_displayed_four | 0.1955 |
| 4 | uniform_sixteen | 0.0625 |
| 4 | slot_rule | 0.3334 |
| 5 | semantic_original | 0.0895 |
| 5 | uniform_displayed_four | 0.2117 |
| 5 | uniform_sixteen | 0.0625 |
| 5 | slot_rule | 0.5858 |

All condition-level fidelity intervals, baseline-assigned original/slot probabilities and discrepancies from observed model probabilities are in [baseline_fidelity.csv](baseline_fidelity.csv). The displayed nine-condition averages above are descriptive; do not average their condition-level interval endpoints to infer a pooled interval.

## Secondary rule and matched-control diagnostics

Forward-π and inverse-π coincide on involutions and on some fixed queries. The following comparison uses only the 14 non-involutions and only cases for which the two predicted values differ. Eligible outcomes are averaged within each dictionary before resampling, preserving row clustering. These are descriptive secondary diagnostics with no additional gates.

| Seed | Inverse answer fidelity [95% CI] | Forward answer fidelity [95% CI] | Inverse−forward fidelity [95% CI] | Inverse−forward target P [95% CI] |
| --- | --- | --- | --- | --- |
| 0 | 0.5450 [0.5305, 0.5590] | 0.0937 [0.0884, 0.0993] | 0.4514 [0.4342, 0.4683] | 0.4394 [0.4237, 0.4547] |
| 1 | 0.5812 [0.5684, 0.5934] | 0.1093 [0.1039, 0.1143] | 0.4720 [0.4558, 0.4880] | 0.4666 [0.4512, 0.4818] |
| 2 | 0.2839 [0.2704, 0.2984] | 0.1675 [0.1608, 0.1741] | 0.1164 [0.1001, 0.1343] | 0.1094 [0.0941, 0.1255] |
| 3 | 0.3857 [0.3753, 0.3963] | 0.1693 [0.1636, 0.1748] | 0.2164 [0.2026, 0.2308] | 0.2134 [0.2006, 0.2266] |
| 4 | 0.3340 [0.3214, 0.3472] | 0.1508 [0.1447, 0.1569] | 0.1833 [0.1676, 0.1984] | 0.1746 [0.1602, 0.1887] |
| 5 | 0.5857 [0.5746, 0.5965] | 0.0861 [0.0812, 0.0909] | 0.4996 [0.4860, 0.5126] | 0.4903 [0.4775, 0.5028] |

| Seed | Canonical−native grouped original P [95% CI] | Value−coherent P on the SAME wrong target [95% CI] | Other two wrong displayed targets, mean P [95% CI] |
| --- | --- | --- | --- |
| 0 | 0.3282 [0.3066, 0.3487] | 0.4220 [0.4092, 0.4349] | 0.0934 [0.0892, 0.0976] |
| 1 | 0.3905 [0.3708, 0.4098] | 0.4626 [0.4504, 0.4745] | 0.1080 [0.1035, 0.1124] |
| 2 | 0.0735 [0.0522, 0.0942] | 0.1280 [0.1159, 0.1390] | 0.1673 [0.1618, 0.1728] |
| 3 | 0.1824 [0.1629, 0.2016] | 0.1973 [0.1867, 0.2082] | 0.1668 [0.1622, 0.1712] |
| 4 | 0.1334 [0.1120, 0.1560] | 0.1741 [0.1627, 0.1854] | 0.1505 [0.1457, 0.1555] |
| 5 | 0.3877 [0.3670, 0.4074] | 0.4814 [0.4706, 0.4922] | 0.0854 [0.0815, 0.0893] |

The matched wrong-target contrast uses the same inverse-π target in each value/coherent derangement pair. It distinguishes specific redirection from an indiscriminate increase in wrong answers. All 23 per-permutation coherent-minus-value original-probability contrasts and intervals are retained in scores.json, as are the individual non-involution comparisons and query breakdowns.

## Descriptive attention diagnostics

All four heads are retained and equally weighted; none was selected using outcomes. Layer 1 statistics average each head's attention over all four value positions to either the true key or the coordinate-associated key. Layer 2 statistics use query attention to the original or coordinate-implied value. Each table cell is the four head means (heads 0,1,2,3), averaged over the nine maps in that family. These summaries do not establish attention causality or serve as extra success criteria.

| Seed | Family | L1 true-key heads | L1 slot-key heads | L2 original-value heads | L2 slot-value heads |
| --- | --- | --- | --- | --- | --- |
| 0 | value_derangement | 0.233, 0.202, 0.176, 0.136 | 0.300, 0.337, 0.376, 0.583 | 0.087, 0.091, 0.078, 0.090 | 0.387, 0.385, 0.390, 0.389 |
| 0 | coherent_control | 0.300, 0.337, 0.377, 0.583 | 0.300, 0.337, 0.377, 0.583 | 0.362, 0.356, 0.366, 0.369 | 0.362, 0.356, 0.366, 0.369 |
| 1 | value_derangement | 0.147, 0.163, 0.139, 0.172 | 0.555, 0.506, 0.551, 0.387 | 0.106, 0.114, 0.108, 0.104 | 0.484, 0.460, 0.487, 0.485 |
| 1 | coherent_control | 0.553, 0.505, 0.551, 0.387 | 0.553, 0.505, 0.551, 0.387 | 0.472, 0.447, 0.483, 0.474 | 0.472, 0.447, 0.483, 0.474 |
| 2 | value_derangement | 0.227, 0.230, 0.243, 0.225 | 0.311, 0.309, 0.270, 0.224 | 0.137, 0.134, 0.134, 0.129 | 0.190, 0.179, 0.198, 0.204 |
| 2 | coherent_control | 0.311, 0.311, 0.271, 0.224 | 0.311, 0.311, 0.271, 0.224 | 0.212, 0.193, 0.216, 0.227 | 0.212, 0.193, 0.216, 0.227 |
| 3 | value_derangement | 0.200, 0.195, 0.203, 0.168 | 0.400, 0.402, 0.314, 0.482 | 0.137, 0.146, 0.135, 0.139 | 0.324, 0.323, 0.333, 0.332 |
| 3 | coherent_control | 0.401, 0.404, 0.314, 0.483 | 0.401, 0.404, 0.314, 0.483 | 0.338, 0.340, 0.350, 0.349 | 0.338, 0.340, 0.350, 0.349 |
| 4 | value_derangement | 0.194, 0.204, 0.228, 0.200 | 0.225, 0.310, 0.318, 0.224 | 0.131, 0.129, 0.133, 0.138 | 0.242, 0.247, 0.237, 0.237 |
| 4 | coherent_control | 0.225, 0.309, 0.317, 0.223 | 0.225, 0.309, 0.317, 0.223 | 0.237, 0.242, 0.236, 0.236 | 0.237, 0.242, 0.236, 0.236 |
| 5 | value_derangement | 0.187, 0.169, 0.194, 0.149 | 0.304, 0.491, 0.408, 0.526 | 0.089, 0.079, 0.097, 0.096 | 0.429, 0.458, 0.417, 0.408 |
| 5 | coherent_control | 0.305, 0.491, 0.408, 0.527 | 0.305, 0.491, 0.408, 0.527 | 0.434, 0.460, 0.423, 0.414 | 0.434, 0.460, 0.423, 0.414 |

All individual condition/head means are in [attention_diagnostics.csv](attention_diagnostics.csv); per-query head means remain in scores.json. Full class probabilities and per-case attention summaries remain in the raw NPZ files.

## Limits and uncertainty

All mean intervals use 2,000 query-stratified dictionary-row bootstraps with fixed seed 7200001. The same weights are shared across conditions and six model seeds. Primary margin aggregation averages the nine derangements within each row before resampling. These intervals are conditional on these six frozen models, task distribution and coordinate interventions; they do not quantify variation over a population of training seeds or tasks. Individual condition intervals are not simultaneous.

External positional reassignment is an intervention, not spontaneous layout generalization. The grouped layout changes causal predecessor sets as well as adjacency relative to interleaved training; canonical coordinates do not restore the original intermediate attention mask. A successful specific redirection supports causal control of binding by trained positional coordinates in this setting. It does not establish a unique complete algorithm, strict necessity of one head, arbitrary-layout competence, or novelty over all prior literature. Failure of any registered gate must remain visible even if other diagnostics look favorable.

All computation was local CPU execution; no inference API or new training was used. All E3 artifacts are namespaced under experiment3/ and outputs/experiment3/. Frozen E1/E2 sources and results remain unchanged.

## Reproduction

The public registration identifies exact inputs, weights, predictions and scoring code. Regenerate only the report from existing outcomes with the command below. The score command reads saved arrays and also performs no model forward. Archive any existing generated score/report files before a deliberate rerun if their original runtime record must be retained.

```sh
.venv/bin/python -m experiment3.score
.venv/bin/python -m experiment3.report
```
