# Frozen discovery-guided prediction

The forecast is that these three trained models route primarily through key-dependent information at value positions: layer 1 builds a contextual representation at each value, layer 2 compares the final query with keys projected from those positions, and values projected there deliver answer content. This is an approximate causal account of the tested pathways, not proof that any vector literally encodes a human-readable key, that a unique circuit has been recovered, or that the pathway is necessary in the unperturbed computation.

## Evidence available before prediction

Only the model/evaluator/scorer/protocol source, training metadata and recovery account, discovery summary, and prospective review documents were used. No checkpoint was run by the predictor; no additional interventions or confirmatory outputs were inspected. Blinding is procedural on a shared filesystem, without operating-system access isolation. The coordinator reports that confirmation has not yet been generated. Key-writing and query-positional-pointer explanations were supplied in advance, so this is hypothesis discrimination and prediction, not an independent invention of those hypotheses.

In the 1,024 discovery cases, all three final seeds had 100% recipient and active donor accuracy. A key-swap donor changed the answer without moving value tokens: transplanting L1 value-position residuals or L2 keys at those positions nearly completely transferred the donor answer. Transplanting the L1 query residual or L2 query did not. Value-swap donors instead transferred through L2 values and L1 value-position residuals. The largest key-swap L2-value effect was approximately 0.00063; this small leakage makes a perfectly isolated symbolic decomposition an approximation. Matched swaps and wrong-position effects were small. Last-layer nonquery Q is a structural zero and cannot distinguish mechanisms.

Training selection is exploratory: the initial embedding SD 0.02 runs stalled near the 25% displayed-value baseline, and the final SD 1.0 setting was selected after pilot runs. Final seeds 0, 1, and 2 each trained for 2,000 steps on the fixed schedule. None is removed based on interventions. The 5,120-example reused training-validation set is not confirmation.

## Numeric forecast rule

For each seed, define `D` as the discovery mean full-donor effect averaged over active key and active value swaps. The frozen scales are **1.998417616**, **1.998462737**, and **1.998595357** for seeds 0, 1, and 2 respectively. The authoritative values are in `predictions.json`. This one scale per model captures finite answer confidence; individual site means are not fitted. All other parameters are binary mediator-inclusion rules. Every seed has all 53 condition forecasts, for 159 numbers total.

| Intervention | Active key swap | Active value swap | Either matched swap |
|---|---:|---:|---:|
| L1 residual at exchanged value positions | D | D | 0 |
| L1 residual at final query | 0 | 0 | 0 |
| L2 K at exchanged value positions | D | 0 | 0 |
| L2 Q at final query | 0 | 0 | 0 |
| L2 V at exchanged value positions | 0 | D | 0 |
| L2 K at exchanged key positions | 0 | 0 | 0 |
| L2 Q at exchanged value positions | 0 | 0 | 0 |
| L1 residual at complementary value positions | 0 | 0 | 0 |
| L2 K at complementary value positions | 0 | 0 | 0 |
| **Unseen joint L2 K + Q** | D | 0 | 0 |
| **Unseen joint L2 K + V** | D | D | 0 |
| **Unseen joint L2 Q + V** | 0 | D | 0 |
| **Unseen joint L1 values + query residuals** | D | D | 0 |

Same-donor combinations use untouched activations from the same donor, transplanted simultaneously. The model predicts transfer if the appropriate causal channel is included and no additional transfer when an inactive channel is added. It does not sum transplant effects or estimate interaction coefficients. The four same-donor joint interventions have never been evaluated in discovery. One additional, stronger composition test was selected prospectively **after discovery and before any confirmation**: `cross/keyK_valueV`. It simultaneously patches L2 K at the two exchanged value positions from the active **key-swap donor** and L2 V at those positions from the active **value-swap donor**. It is the sole exception to using a single donor for both components. The key routing selects the other position, whose transplanted content is now the original answer. The frozen expected effect is therefore **0.0 for every seed**, with bounds **[-0.15, 0.15]** and an additional **original-answer accuracy >=95%** gate. An untouched donor that swaps both keys and values preserves the original association and supplies the full-donor comparison. This cross intervention has not been run in discovery. Its additive discovery baseline sums `key/l2_k_values` and `value/l2_v_values`, clipped to [-2, 2], predicting approximately 2 instead of 0. This tests a compositional prediction where empirical addition and the explanation materially disagree.

All cases in confirmation are also new relative to the audited streams, subject to the prospective overlap gate.

The forecast tolerance is absolute error **0.15** on each condition mean. Transfer bounds are `[D - 0.15, min(2, D + 0.15)]`; null bounds are `[-0.15, 0.15]`. These are prespecified practical acceptance bounds, not predictive intervals with claimed coverage. On a near-deterministic two-answer switch, 0.15 is approximately 7.5% of the full scale of 2. It is deliberately larger than discovery sampling uncertainty to test useful mechanistic prediction rather than rounded numerical equality.

## Decisions and alternatives

The primary mechanism contrast is **key/L2 K at value positions minus key/L2 Q at the query**. Support for substantial K-mediated routing requires the paired-bootstrap 95% lower bound to exceed **1.5** in every seed, with recipient and both active unpatched donor accuracies at least **95%** and all no-op maximum logit errors at most **1e-6**. The 1.5 threshold demands over three quarters of the maximum contrast scale of 2. Confirmation of this full quantitative account additionally requires every one of the 53 condition forecasts to be within 0.15 in each seed and cross/keyK_valueV original-answer accuracy at least 95%. Always report the count and identity of misses; a primary-contrast success alone is narrower evidence than complete forecast success. No family, seed, or failed example is discarded.

A large Q effect and small K effect would favor the supplied query-pointer alternative; large effects at both sites would favor redundant or mixed routing. Seed disagreement would establish heterogeneity rather than a universal mechanism. Strong wrong-position or matched effects would challenge locality or transplantation validity. Failure of value/V transfer would challenge the content pathway. A joint patch far outside its bound despite accurate constituent forecasts would falsify the proposed compositional rule and suggest interaction or incompatible representations. Low task accuracy or failed no-op checks makes the mechanism test inconclusive, with the failure still reported.

Alternatives that remain compatible with a positive result include distributed contextual codes rather than literal copied key identities, redundant pathways not revealed by sufficiency patches, and unnatural donor-recipient activation interactions. Head-specific or token-specific mechanisms are not identified by whole-vector patches. Matched controls move different positions from active swaps, and the alternative-minus-original contrast can miss redistribution among other answer classes; original/alternative answer rates remain visible.

The empirical discovery-mean baseline is a strong comparator on familiar and same-donor conditions. It sums constituent discovery effects for novel combinations, clipped to [-2, 2]. The four same-donor combinations usually contain one effective channel, so their empirical forecasts closely resemble this mechanistic rule. The cross-donor composition deliberately separates the two forecasts, but remains a single post-discovery chosen test and does not by itself establish general predictive superiority. Beating zero and full-donor baselines would establish selectivity; it would not establish added predictive value beyond empirical calibration. Claim added forecast value over the empirical baseline only if the paired 95% interval for AI-minus-empirical mean MSE is wholly below zero in every seed. Otherwise report no demonstrated advantage, including when all practical forecast bounds pass. Three models support a local pilot conclusion, not arbitrary-model interpretability or general AI self-understanding.
