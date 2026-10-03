# Experiment 6: blinded forecasts for pretrained GPT-2

Prepared 2026-10-03 before any shifted-layout or repaired GPT-2 confirmation. The fixed checkpoint is `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The selected format is `one_demo`, and the confirmation sample is 512 fresh dictionaries, balanced at 128 queries per pair position. This is one pretrained checkpoint, not six new model replications.

## Information available to the predictor

The predictor read only the E6 calibration plan, input/format definitions, condition definitions, and earlier E1–E5 evidence. The coordinating role supplied the selected format name and sample size. The predictor has not seen E6 calibration scores, checkpoint metrics, calibration logs, individual inputs, or confirmatory outcomes and has performed no E6 model forwards. Format selection used the separately declared bounded native-only procedure. Its outcome is limited information available to this forecast; it does not imply that the chosen format passed the competence threshold.

The files are frozen before confirmation. The predictor's access restriction is procedural on a shared filesystem, not enforced isolation. No numerical forecast may be revised after observing confirmation.

## Substantive hypothesis

The unchanged visibility guard at block 0 is unlikely to restore near-native final answers in this pretrained twelve-block model. It may yield a modest benefit. Under canonical coordinates, the correct guard restores first-block value-chunk states and leaves query-suffix states equal to native, up to numerical error. This is an implementation identity, not evidence that the final output must recover.

In the earlier two-layer architecture, only one subsequent attention block separated the restored states from the answer. Here eleven blocks remain. Changed key-chunk states can alter later states, including value and query representations. The model was also pretrained on ordinary text rather than trained specifically on this four-pair task. Canonical coordinate reassignment preserves token embeddings, but it does not establish that the model treats the altered serialization as a familiar lookup problem.

Applying the same mask at block 5 or all twelve blocks might restrict disruptive information closer to the eventual answer computation. It might instead remove useful context. Neither intervention has a general native-state restoration identity beyond the first block, and neither was selected from repaired-model outcomes. Their numerical forecasts express a tentative ordering, not a claim that later masking is known to be better. Secondary success cannot retroactively validate the block-0 explanation.

## Frozen numerical forecasts

All accuracy forecasts concern choosing the highest-scoring token among the 16 permitted answer tokens (numbers 20–35). They are **not unrestricted next-token generation accuracies**. For a case with target y, define conditional target probability as

\[
p_{16}(y)=\frac{p(y)}{\sum_{v=20}^{35}p(v)},
\]

using the model's ordinary next-token distribution before conditioning. The contrast below is the casewise block-0 guard probability minus the equal mean of the eight block-0 sham probabilities, averaged over the same 512 cases. It is not a contrast against the best sham, raw target probability, or accuracy.

| Endpoint | Point forecast | Subjective plausibility interval |
| --- | ---: | ---: |
| `native`: 16-choice accuracy | .85 | [.60, .99] |
| `grouped_canonical`: 16-choice accuracy | .45 | [.20, .80] |
| `guard_block0`: 16-choice accuracy | .50 | [.20, .85] |
| `guard_block0` minus mean eight shams: conditional target probability | .02 | [−.08, .12] |
| `guard_block5`: 16-choice accuracy | .55 | [.20, .90] |
| `guard_all`: 16-choice accuracy | .65 | [.25, .95] |

These are judgmental forecasts. They use no E6 fitted parameters or model-specific scores. The intervals express broad subjective plausibility; they have no calibrated or asserted nominal coverage. Their width reflects uncertainty about native competence, the effect of grouped serialization, and where the relevant computation occurs. They are not sampling confidence intervals and do not justify exclusions or adaptive extensions.

The forecast rule is deliberately coarse: anticipate reasonably useful native lookup after bounded format selection, substantial impairment from canonical grouping, a small block-0 benefit, a small average advantage over matched masks, and possible additional benefit from later or repeated masking. The earlier toy-model effect sizes are not copied into this model. No numerical forecasts are made for each individual sham, grouped physical positions, raw target probability, answer-token mass, or unrestricted generation; all these outcomes must still be reported.

## Evaluation and falsification

Compare each of the six observed aggregates with its point forecast using absolute-error tolerance .10. Report every error and miss, the six-endpoint MAE, and the subjective intervals without relabeling them as confidence intervals. This descriptive forecast score does not determine causal success. The separate confirmatory protocol fixes competence, validity, recovery, specificity, and secondary decisions.

Near-native recovery under `guard_block0`, with a clear advantage over matched controls on a competent native task, would contradict the substantive expectation of limited first-block transfer. Failure of the native task would instead limit the repair test's interpretability; it would not establish that the guard failed to repair a learned lookup computation. A block-0 benefit indistinguishable from the mean of shams would weaken specificity even if accuracy increased. Strong secondary recovery would motivate a different layer-dependent explanation, not rescue the original primary claim.

Uniform selection among all 16 answer tokens gives 6.25% accuracy; uniform selection among the four displayed values gives 25%. These reference rates differ from unrestricted vocabulary generation. All conditions use the known pair mapping and fixed token occurrences. No model weights are changed. Newly generated combinations do not establish absence from GPT-2's unavailable pretraining corpus. A result in this single checkpoint cannot establish general transfer to pretrained language models.
