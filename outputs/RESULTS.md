# Experiment 1: confirmatory v2 results

**All preregistered v2 decision gates passed in all three models.** The original v1 attempt aborted at its input-overlap gate before any model evaluation and remains an aborted test. V2 was separately registered with unchanged forecasts and an exact disjoint sample.

This supports a discovery-guided causal prediction for these specific small models and interventions. It does not establish a unique internal algorithm or general AI self-understanding.

## Registration and execution

- [Original registration](https://github.com/posix4e/ai-understanding-ai/commit/e3ea745929e44ed594667ce68330e7eda8b16acd): verified 2026-10-03 12:40:52 UTC; aborted before model forwards on four training-overlap incidences.
- [V2 registration](https://github.com/posix4e/ai-understanding-ai/commit/b2d3bdd55db2dd9a8f2f3f2548dabe9fe8f53b37): all 45 frozen file identities remotely verified 12:45:04 UTC; model-run gate entered 12:48:07 UTC.
- V2 froze 2,048 recipient bundles and all five donor input arrays before evaluation. Four of 2,052 candidates were rejected by identity only. Rebuilt forbidden set: 2,057,974 unique inputs; zero accepted overlaps and zero duplicate recipients.
- All training and experiment execution used the local Apple M3 CPU, 16 GiB memory, four threads, PyTorch 2.14.1. No TDX, cloud training, or paid inference API. Separate Astra implementer, prediction and skeptical-review roles worked in Codex; role inference itself was not locally hosted.
- Model: 70,720 parameters, two causal transformer layers, four heads, width 64, MLP 128. Three seeds; fixed 2,000 updates with embedding SD 1.0 after disclosed exploratory recovery.

## Task and primary mechanism test

Each model answered 2,048/2,048 held-out recipient lookups correctly (100%; per-model Wilson 95% interval 99.813–100%). All query-position groups and both active unpatched donor families also had 100% accuracy. Chance is 6.25% over the full output vocabulary; choosing a displayed value gives 25%.

The effect is the change in P(alternative) − P(original), patched minus recipient, on a [-2,2] scale. The registered primary contrast compares key-swap L2 K patches at value positions against L2 Q patches at the final query.

| Seed | K-minus-Q contrast | Paired bootstrap 95% CI | Forecasts within 0.15 | Largest mean forecast error |
|---|---:|---|---:|---:|
| 0 | 1.998133 | [1.998100, 1.998163] | 53/53 | 0.000555 |
| 1 | 1.997834 | [1.997711, 1.997919] | 53/53 | 0.002686 |
| 2 | 1.998322 | [1.998285, 1.998355] | 53/53 | 0.001161 |

Every primary lower confidence bound exceeds the frozen threshold 1.5. No-op maximum logit differences were exactly zero at every recorded site. Wrong-position, wrong-component and matched intervention forecasts all met their bounds. The last-layer nonquery Q control is structurally zero, so it tests implementation locality rather than distinguishing mechanisms.

The selective transfers favor key-dependent routing information at value positions: key-swap L2 K patches redirect retrieval; value-swap L2 V patches replace content; query Q transfers have near-zero effects. Whole-vector transplantation establishes sufficiency under these interventions, not literal key-copy semantics or necessity.

## New cross-donor composition test

L2 K came from a key-swapped dictionary; L2 V came from a value-swapped dictionary. The explanation predicted that swapping both routing and content restores the original answer, giving effect 0. The additive discovery baseline predicted effect 2. This intervention was selected after discovery but evaluated only after registration.

| Seed | Predicted effect | Observed effect (descriptive 95% CI) | Original-answer accuracy |
|---|---:|---|---:|
| 0 | 0 | 0.000555 [0.000300, 0.000811] | 100.0000% |
| 1 | 0 | 0.001385 [0.000764, 0.002007] | 99.9512% |
| 2 | 0 | 0.001161 [0.000199, 0.002123] | 99.9512% |

Cross-donor transplantation produced one incorrect answer in seed 1 and one in seed 2; both were other value classes, not the alternative target. Value-swap L2 V transfer also missed the donor answer on four seed-1 cases and one seed-2 case. These failures are retained. Small nonzero mean effects also reject exact numerical invariance; they remain far inside the prespecified practical tolerance.

## Forecast scoring

Mean squared error weights all 53 condition means equally. The empirical baseline uses discovery means for familiar conditions and clipped sums for novel combinations. The full-donor benchmark uses observed donor behavior; it is not an ex ante numeric forecast.

| Seed | AI explanation | Empirical/additive | Zero effect | Full-donor transfer |
|---|---:|---:|---:|---:|
| 0 | 1.4723381e-08 | 0.075429782 | 0.75346737 | 1.2055525 |
| 1 | 3.4550036e-07 | 0.075367425 | 0.75305227 | 1.2055193 |
| 2 | 4.7371222e-08 | 0.075384122 | 0.75351439 | 1.2057621 |

**The overall advantage over the empirical/additive baseline is driven by the single cross-donor cancellation test.** On the 36 familiar conditions, the empirical discovery-mean baseline had lower MSE in every seed. The familiar gaps are small but their paired bootstrap intervals favor the empirical baseline. The skeptical reviewer's explicitly post hoc diagnostic excluding the cross condition also reverses the overall advantage in every seed. This is evidence for one compositional generalization, not broad superiority over empirical prediction. The other 16 novel combinations largely combine an active and an inactive pathway. See [the independent review](SKEPTICAL_REVIEW.md).

Paired 2,000-replicate bootstrap intervals for overall AI-minus-additive MSE are entirely below zero in all seeds; novel-group intervals also favor the explanation. Full scores, intervals, individual-case MSE, and subgroup comparisons are in [scores.json](confirmatory_v2/scores.json). Every mean forecast and result is in [forecasts_vs_results.csv](forecasts_vs_results.csv).

## Limits and negative results

- Initial small-embedding training failed near 25% across three seeds; longer training and increased learning rates also failed. The successful initialization was chosen using reused pilot validation, not prospectively at project start.
- The original preregistered attempt failed its disjointness gate. V2 changes the input distribution by a disclosed identity-based selection rule. No outcome was available when this change or forecast reaffirmation occurred.
- Mechanisms were suggested in advance; this is not independent discovery. Blinding was procedural on shared storage, not enforced by access controls.
- All features and heads at chosen sites were patched together. Mixed, redundant or distributed codes remain compatible with the result. Matched swaps change different positions than active swaps.
- Condition means can hide individual-case errors. Probability contrasts can hide movement into third answers, as the two cross-test mistakes demonstrate.
- Confidence intervals condition on trained models and discovery calibration. Inputs are shared across only three seeds; there is no population-level across-training-seed confidence interval or multiple-testing claim over 159 cells.
- Synthetic fixed-length dictionaries and a tiny two-layer network do not establish how learning works or explain frontier models.

## Next experiment

Preregister head-specific pathway ablations and rescue patches, with additional training seeds and dictionary layouts whose key/value distances vary. Use a broader factorial family of independently sourced routing and content donors and compare against a symbolic binding baseline, not only additive effects. Test whether the same explanation predicts necessity, interference and layout generalization. Do not tune the frozen explanation on those outcomes.
