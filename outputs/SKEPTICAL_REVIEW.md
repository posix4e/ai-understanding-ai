# Independent skeptical review of the released v2 results

**The amended v2 experiment passes its frozen quantitative gates.** It supports a useful approximate routing/content explanation for these three trained models and this intervention family. It does not establish unique circuit recovery, necessity, general mechanistic understanding, or broad superiority over empirical prediction. The original v1 experiment aborted at its input-overlap gate and must remain reported as aborted.

## Integrity and independent checks

The v2 lock identifies commit `b2d3bdd55db2dd9a8f2f3f2548dabe9fe8f53b37`, with recorded remote verification at 12:45:04 UTC on October 3, 2026. The confirmatory start record names the same commit at 12:48:07 UTC. I independently checked that all **45 frozen file hashes still match** the lock. These checks verify consistency with the recorded remote readback, not a second independent network readback.

All three raw result files contain the exact frozen 2,048 recipients, query indices and targets. I checked finite values and recomputed all 53 condition means per seed directly from stored case-level probability contrasts, without calling the experiment runner or scorer. I independently recalculated mean-squared errors and reproduced the paired-bootstrap primary contrast and overall AI-minus-empirical loss intervals. Double-precision accumulation produces tiny differences from the stored float32 summaries; none approaches a decision threshold or changes a finding.

The prior input-only review reconstructed the 2,057,974-input forbidden set and reproduced the selected arrays: four whole groups rejected from 2,052 candidates, leaving 2,048, with zero forbidden matches or duplicate recipients. The evaluator's released pre-forward audit confirms the same identities. This was an explicitly amended sampling procedure; it was not an exclusion allowed by the original failed v1 registration.

## Gates and main findings

| Measure | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Recipient / active donor accuracy | 100% / 100% | 100% / 100% | 100% / 100% |
| Maximum reported no-op logit error | 0 | 0 | 0 |
| Condition means within frozen absolute error 0.15 | 53/53 | 53/53 | 53/53 |
| Largest absolute forecast error | 0.000555 | 0.002686 | 0.001161 |
| Key-donor L2 K-minus-Q mean effect | 1.998133 | 1.997834 | 1.998321 |
| Recomputed paired 95% interval | [1.998100, 1.998163] | [1.997711, 1.997919] | [1.998285, 1.998355] |
| Cross-donor effect, forecast 0 | 0.000555 | 0.001385 | 0.001161 |
| Cross-donor original answer retained | 2048/2048 | 2047/2048 | 2047/2048 |

All primary-contrast lower bounds exceed 1.5, all accuracy gates exceed 95%, and all **159 condition-mean forecasts** pass. Baseline accuracy is also 100% at each queried pair position. The largest error is seed 1's value-donor L2 V transplant. Key-donor L2 K transfer is near complete while query-Q transfer is tiny; value-donor V transfer supplies the complementary content result. The cross-donor composition retains the original answer almost always, as the explanation predicted.

## Actual negative outcomes and the baseline comparison

**The empirical discovery baseline is more accurate on familiar interventions in every seed.** The frozen familiar-group bootstrap intervals for AI-minus-empirical MSE are wholly positive. Its numerical advantage is small, but the direction is consistent and should not be hidden by the overall score.

| Condition-mean MSE | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| AI, all 53 conditions | 1.47e-8 | 3.46e-7 | 4.73e-8 |
| Empirical/additive, all 53 | 0.07543 | 0.07537 | 0.07538 |
| AI, 36 familiar conditions | 6.56e-9 | 2.28e-7 | 1.60e-8 |
| Empirical, 36 familiar | 1.03e-9 | 1.78e-7 | 7.38e-9 |

The AI wins the preregistered overall comparison in all seeds: its paired loss-gap intervals are wholly below zero. **That advantage comes from the single cross-donor condition.** The additive baseline predicts approximately 2 there; the observed effect is approximately zero. As an explicitly post-hoc diagnostic, removing that one condition makes the empirical/additive baseline better over the remaining 52 conditions in every seed. This does not replace the primary score; it identifies its source.

The cross test is a meaningful prospective success, but it was deliberately chosen after discovery to separate cancellation from addition. The other 16 novel conditions mostly add inactive channels to an effective one and give very similar mechanistic and empirical forecasts. The experiment does not compare the explanation against a strong alternative nonlinear composition model. General predictive superiority is therefore unsupported.

**Mean-forecast success is not perfect case-level prediction.** Cross-donor cancellation fails to retain the original answer on one case in each of seeds 1 and 2. Value-donor V transfer selects the intended alternative on 2,044/2,048 cases for seed 1 and 2,047/2,048 for seed 2. Small effects in nominally inactive pathways also remain. These are compatible with the preregistered approximate mean account, but contradict any claim of an exact symbolic decomposition or universal per-example behavior.

## Remaining interpretation limits

* Transplantation shows that these donor-dependent activations can mediate behavioral changes. It does not establish necessity, identify a uniquely meaningful key representation, exclude redundant routes, or recover every computation. Whole-vector swaps can also produce unnatural activation combinations.
* The hypotheses were supplied in advance. The AI used discovery intervention results to choose its account; this is hypothesis discrimination and prospective prediction, not independent invention of the mechanism. Blinding was procedural on a shared filesystem, not access-enforced.
* Three selected, successfully trained models from one small architecture and task are a local replication. Training-setting exploration preceded the final fixed schedule. Shared inputs and conditionally bootstrapped cases do not measure uncertainty across arbitrary training runs or architectures.
* The 159 numbers are correlated forecasts, including controls that are structurally zero. They are not 159 independent demonstrations. Intervals condition on discovery calibration and checkpoints; they are not simultaneous bounds across all cells.
* Matched donors alter different positions from active donors. The two-target probability contrast can omit redistribution among other outputs. Original-answer rates mitigate this but do not establish a complete account of the output distribution.
* The 0.15 acceptance tolerance is practical, not a calibrated predictive interval. Passing it establishes the declared level of usefulness, not exact numerical theory. The input-filter amendment and v1 failure remain part of the evidence record.

The warranted conclusion is that a discovery-guided explanation made accurate frozen **condition-mean predictions**, including one discriminating cross-donor composition, for the specified interventions in three tiny transformers. The clearest next evidential step would be a separately preregistered family of diverse composition tests against stronger composition baselines and additional trained models.
