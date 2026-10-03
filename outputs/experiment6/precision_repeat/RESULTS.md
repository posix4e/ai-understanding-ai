# E6: GPT-2 precision repeat on reused inputs

**Classification: repair_failed.** Native competence and ordering damage were established, but the primary repair did not satisfy every registered criterion.

The same 512 dictionaries as the original run; one pretrained checkpoint. This is an outcome-informed numerical repeat, not independent confirmation. Restricted accuracy ranks 16 answer tokens and is not unrestricted generation accuracy.

| Condition | Restricted accuracy | Unrestricted next-token accuracy | Raw target probability | Conditional target probability | Candidate mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 0.931641 | 0.869141 | 0.202979 | 0.357649 | 0.553946 |
| grouped_physical | 0.097656 | 0.000000 | 0.005205 | 0.071044 | 0.122788 |
| grouped_canonical | 0.103516 | 0.000000 | 0.005116 | 0.074250 | 0.107423 |
| grouped_noop | 0.103516 | 0.000000 | 0.005116 | 0.074250 | 0.107423 |
| guard_block0 | 0.095703 | 0.000000 | 0.004959 | 0.070108 | 0.115140 |
| sham_0 | 0.093750 | 0.000000 | 0.004883 | 0.070074 | 0.113583 |
| sham_1 | 0.093750 | 0.000000 | 0.004882 | 0.070074 | 0.113548 |
| sham_2 | 0.097656 | 0.000000 | 0.004851 | 0.070153 | 0.112950 |
| sham_3 | 0.097656 | 0.000000 | 0.004850 | 0.070153 | 0.112914 |
| sham_4 | 0.097656 | 0.000000 | 0.004929 | 0.070194 | 0.114494 |
| sham_5 | 0.099609 | 0.000000 | 0.004820 | 0.069978 | 0.113389 |
| sham_6 | 0.099609 | 0.000000 | 0.004819 | 0.069978 | 0.113352 |
| sham_7 | 0.097656 | 0.000000 | 0.004901 | 0.070020 | 0.114991 |
| guard_block5 | 0.105469 | 0.000000 | 0.005149 | 0.074630 | 0.110241 |
| guard_all | 0.312500 | 0.039062 | 0.017729 | 0.125980 | 0.103971 |

## Registered gates

| Gate | Outcome |
| --- | --- |
| implementation_valid | PASS |
| native_overall_at_least_80_percent | PASS |
| every_native_query_at_least_70_percent | PASS |
| damage_point_at_least_10_percentage_points | PASS |
| damage_ci_lower_above_zero | PASS |
| repair_accuracy_ci_lower_above_5_percentage_points | FAIL |
| native_minus_guard_accuracy_ci_upper_at_most_5_percentage_points | FAIL |
| specificity_conditional_probability_ci_lower_above_02 | FAIL |

## Paired contrasts

| Contrast | Mean | Individual 95% interval |
| --- | ---: | --- |
| damage_native_minus_grouped_accuracy | 0.828125 | [0.794922, 0.859375] |
| repair_guard_minus_grouped_accuracy | -0.007812 | [-0.019531, 0.003906] |
| deficit_native_minus_guard_accuracy | 0.835938 | [0.802734, 0.867188] |
| specificity_guard_minus_mean_shams_conditional_probability | 0.000030 | [-0.000022, 0.000083] |
| guard_minus_grouped_raw_target_probability | -0.000157 | [-0.000252, -0.000059] |
| guard_minus_grouped_candidate_mass | 0.007717 | [0.006255, 0.009092] |
| guard_minus_best_point_sham_conditional_probability_descriptive | -0.000086 | Descriptive; selected by point estimate |
| secondary_guard_block5_minus_grouped_accuracy | 0.001953 | [-0.003906, 0.007861] |
| secondary_guard_all_minus_grouped_accuracy | 0.208984 | [0.181641, 0.234424] |

## Query-position results

| Condition | Query position | N | Restricted accuracy | Unrestricted accuracy | Conditional target probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 0 | 128 | 1.000000 | 1.000000 | 0.437257 |
| native | 1 | 128 | 0.937500 | 0.906250 | 0.355843 |
| native | 2 | 128 | 0.937500 | 0.875000 | 0.334867 |
| native | 3 | 128 | 0.851562 | 0.695312 | 0.302628 |
| grouped_physical | 0 | 128 | 0.187500 | 0.000000 | 0.091991 |
| grouped_physical | 1 | 128 | 0.187500 | 0.000000 | 0.098344 |
| grouped_physical | 2 | 128 | 0.015625 | 0.000000 | 0.059219 |
| grouped_physical | 3 | 128 | 0.000000 | 0.000000 | 0.034621 |
| grouped_canonical | 0 | 128 | 0.195312 | 0.000000 | 0.094863 |
| grouped_canonical | 1 | 128 | 0.203125 | 0.000000 | 0.102809 |
| grouped_canonical | 2 | 128 | 0.015625 | 0.000000 | 0.061010 |
| grouped_canonical | 3 | 128 | 0.000000 | 0.000000 | 0.038316 |
| grouped_noop | 0 | 128 | 0.195312 | 0.000000 | 0.094863 |
| grouped_noop | 1 | 128 | 0.203125 | 0.000000 | 0.102809 |
| grouped_noop | 2 | 128 | 0.015625 | 0.000000 | 0.061010 |
| grouped_noop | 3 | 128 | 0.000000 | 0.000000 | 0.038316 |
| guard_block0 | 0 | 128 | 0.179688 | 0.000000 | 0.089283 |
| guard_block0 | 1 | 128 | 0.179688 | 0.000000 | 0.095703 |
| guard_block0 | 2 | 128 | 0.023438 | 0.000000 | 0.059994 |
| guard_block0 | 3 | 128 | 0.000000 | 0.000000 | 0.035452 |
| sham_0 | 0 | 128 | 0.171875 | 0.000000 | 0.088932 |
| sham_0 | 1 | 128 | 0.179688 | 0.000000 | 0.095682 |
| sham_0 | 2 | 128 | 0.023438 | 0.000000 | 0.060261 |
| sham_0 | 3 | 128 | 0.000000 | 0.000000 | 0.035420 |
| sham_1 | 0 | 128 | 0.171875 | 0.000000 | 0.088932 |
| sham_1 | 1 | 128 | 0.179688 | 0.000000 | 0.095674 |
| sham_1 | 2 | 128 | 0.023438 | 0.000000 | 0.060265 |
| sham_1 | 3 | 128 | 0.000000 | 0.000000 | 0.035425 |
| sham_2 | 0 | 128 | 0.171875 | 0.000000 | 0.088773 |
| sham_2 | 1 | 128 | 0.187500 | 0.000000 | 0.096394 |
| sham_2 | 2 | 128 | 0.031250 | 0.000000 | 0.060096 |
| sham_2 | 3 | 128 | 0.000000 | 0.000000 | 0.035349 |
| sham_3 | 0 | 128 | 0.171875 | 0.000000 | 0.088772 |
| sham_3 | 1 | 128 | 0.187500 | 0.000000 | 0.096386 |
| sham_3 | 2 | 128 | 0.031250 | 0.000000 | 0.060100 |
| sham_3 | 3 | 128 | 0.000000 | 0.000000 | 0.035355 |
| sham_4 | 0 | 128 | 0.179688 | 0.000000 | 0.089106 |
| sham_4 | 1 | 128 | 0.187500 | 0.000000 | 0.096409 |
| sham_4 | 2 | 128 | 0.023438 | 0.000000 | 0.059863 |
| sham_4 | 3 | 128 | 0.000000 | 0.000000 | 0.035399 |
| sham_5 | 0 | 128 | 0.179688 | 0.000000 | 0.088877 |
| sham_5 | 1 | 128 | 0.187500 | 0.000000 | 0.096030 |
| sham_5 | 2 | 128 | 0.031250 | 0.000000 | 0.060054 |
| sham_5 | 3 | 128 | 0.000000 | 0.000000 | 0.034952 |
| sham_6 | 0 | 128 | 0.179688 | 0.000000 | 0.088876 |
| sham_6 | 1 | 128 | 0.187500 | 0.000000 | 0.096022 |
| sham_6 | 2 | 128 | 0.031250 | 0.000000 | 0.060054 |
| sham_6 | 3 | 128 | 0.000000 | 0.000000 | 0.034958 |
| sham_7 | 0 | 128 | 0.179688 | 0.000000 | 0.089211 |
| sham_7 | 1 | 128 | 0.187500 | 0.000000 | 0.096030 |
| sham_7 | 2 | 128 | 0.023438 | 0.000000 | 0.059823 |
| sham_7 | 3 | 128 | 0.000000 | 0.000000 | 0.035017 |
| guard_block5 | 0 | 128 | 0.210938 | 0.000000 | 0.097312 |
| guard_block5 | 1 | 128 | 0.195312 | 0.000000 | 0.102987 |
| guard_block5 | 2 | 128 | 0.015625 | 0.000000 | 0.061143 |
| guard_block5 | 3 | 128 | 0.000000 | 0.000000 | 0.037077 |
| guard_all | 0 | 128 | 0.945312 | 0.156250 | 0.267815 |
| guard_all | 1 | 128 | 0.187500 | 0.000000 | 0.097775 |
| guard_all | 2 | 128 | 0.093750 | 0.000000 | 0.075278 |
| guard_all | 3 | 128 | 0.023438 | 0.000000 | 0.063054 |

Intervals use 2,000 paired query-stratified bootstrap draws shared across all conditions (seed 10600003). They are individual intervals, not simultaneous confidence coverage. Per-query Wilson intervals and all saved metrics are in scores.json.

Middle-block and all-block interventions are secondary and cannot rescue a failed primary claim. First-layer state identities are implementation checks, not evidence of final-answer recovery through 12 layers. The mask uses externally supplied pair relationships and position labels; success would not establish spontaneous layout generalization, a unique algorithm, or generalization to other pretrained models.
