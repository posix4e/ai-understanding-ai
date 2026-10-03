# E6 precision repeat: numerical robustness on reused inputs

Prepared after inspecting the completed original GPT-2 confirmation and before any trained-model forward for this precision repeat. This is a disclosed, outcome-informed numerical follow-up. It is **not a second independent confirmation**, a fresh sample, or a replacement for the original result.

## Why a repeat is justified, and what already failed

The original float32 run has the official classification **implementation_invalid** under its frozen rules. Two first-block absolute-state errors exceeded the registered `1e-5` tolerance:

| Check | Original maximum absolute error | Original outcome |
| --- | ---: | --- |
| Guarded value states versus native | 0.0000133514404296875 | FAIL |
| Query-suffix states versus native | 0.000011444091796875 | FAIL |
| Unedited key states versus grouped baseline | 0 | PASS |
| No-op probabilities | 0 | PASS |

All saved outputs were finite, and model-parameter hashes before and after the original run agreed. The small state discrepancies are compatible with floating-point accumulation effects, but that explanation has not yet been established by the repeat. We do not relax the threshold or retrospectively relabel the original run as valid.

The original behavioral results also failed all three primary repair criteria, independently of the validity gate. Restricted 16-choice accuracy was 93.1640625% for native input, 10.3515625% for canonical grouped input, and 9.5703125% for the first-block guard. The guard-minus-grouped accuracy contrast was -0.0078125 with paired 95% interval [-0.01953125, 0.00390625]; the native-minus-guard accuracy deficit was 0.8359375 with interval [0.802734375, 0.8671875]. The guard-minus-mean-shams conditional target-probability contrast was approximately 0.000030, with an interval spanning zero. These observed failures motivate transparent reporting, not a change of hypothesis. The all-block secondary intervention's original accuracy of 31.25% does not satisfy or replace the first-block claim.

The repeat asks whether higher arithmetic precision satisfies the same implementation identities and whether the original behavioral conclusions persist when those checks are met.

## What changes

Only the model computation dtype changes from float32 to float64. Load the same checkpoint weights used in the original run and promote their float32 values exactly to float64; do not fetch a different checkpoint, train, optimize, or otherwise modify parameter values. The checkpoint remains `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

Parameters, floating-point buffers, attention calculations, intermediate activations and final model outputs must use float64 throughout the repeat. Token IDs, position IDs and discrete masks retain their appropriate integer or Boolean types. Audit the wrapper for accidental casts back to float32. Verify that converting each promoted parameter back to float32 exactly reproduces its original value. Hash the promoted parameters before and after inference; those hashes must agree with each other, while their byte hashes will naturally differ from the original float32 tensors.

There is one explicit representation exception: the unchanged, frozen adapter constructs additive attention masks as float32 tensors containing **only exactly representable zero and negative infinity**. Adding these masks to float64 attention scores promotes them exactly to float64; no finite learned value or activation is rounded to float32. Retain this adapter behavior rather than editing a frozen source file. This exception does not authorize finite float32 attention biases, a float32 softmax, or a float32 intermediate activation. The eager attention path must retain `reorder_and_upcast_attn=False`, since the alternative upcast path explicitly computes attention scores in float32.

Use CPU execution, four threads, batch size eight, eager attention, disabled dropout and no KV cache, as in the original run. Preserve existing numerical attention options. Do not add a second computational change while attributing results to precision.

## What stays fixed

Reuse all 512 original confirmation dictionaries, in the same row order, with 128 queries at each pair position. Keep the selected `one_demo` template, tokenizer, exact token chunks, candidate-token order, canonical position IDs and physical causal mask unchanged. No new prompt selection, calibration, model choice, layer search, head selection, example exclusion or input resampling is allowed.

Run all 15 original conditions: `native`, `grouped_physical`, `grouped_canonical`, `grouped_noop`, `guard_block0`, `sham_0` through `sham_7`, `guard_block5` and `guard_all`. Keep all mask definitions unchanged. Secondary conditions remain secondary and cannot rescue the primary first-block hypothesis.

The state tolerance remains `1e-5`; the no-op probability tolerance remains `1e-6`. Preserve the original finite-output, unchanged-parameter and unedited-key checks. The original scorer and these behavioral thresholds are unchanged:

| Criterion | Same repeat requirement |
| --- | --- |
| Native competence | Restricted accuracy at least 0.80 overall and at least 0.70 in every query stratum |
| Ordering damage | Native-minus-canonical-grouped accuracy at least 0.10, with paired 95% lower bound strictly above zero |
| Primary repair improvement | Guard-minus-canonical-grouped accuracy paired 95% lower bound strictly above 0.05 |
| Near-native recovery | Native-minus-guard accuracy paired 95% upper bound at most 0.05 |
| Specificity | Guard-minus-mean-eight-shams conditional target probability paired 95% lower bound strictly above 0.02 |

Use the identical 2,000 query-stratified, paired bootstrap draws with seed `10600003`, shared across all conditions. Use the unchanged dictionary-row unit, conditional normalization, fixed candidate-order argmax tie rule, percentile intervals and decision classification. Do not infer independent replication from repeated intervals on these same rows. Preserve unrestricted next-token accuracy, raw target probabilities, candidate mass and all per-query summaries alongside the restricted-choice score.

## Freeze, execution and provenance

Create a separate precision-run module and configuration. Preserve all original source files, inputs, registration files, raw arrays, scores, validity records and reports unchanged. Store repeat artifacts in a separately named output directory. Before the first trained-model float64 forward, publicly commit and remotely verify a new manifest containing this plan, the new precision wrapper/configuration, unchanged scientific dependencies, the original input hash, checkpoint identity and exact thresholds. Record verification and run-start timestamps.

The new wrapper may reuse the frozen runner and scorer without editing them. It must pass the separate repeat output directory explicitly, avoiding writes to the original confirmation directory or original RESULTS.md. Record raw-array hashes, source hashes, model dtype, before/after parameter hashes, software versions and completion status. Any interruption or numerical error must be retained rather than silently restarting with changed scientific settings.

## Required reporting and interpretation

Report every repeat condition, every validity check and every failed behavioral gate, regardless of the outcome. Put the original and repeat classifications next to each other. For each of the 15 conditions, report changes in restricted and unrestricted accuracy, raw and conditional target probability, candidate mass, and the number of rows whose restricted or unrestricted prediction changed. Also report the maximum absolute change in saved candidate probabilities and the maximum state-identity errors. These comparisons are descriptive numerical sensitivity checks on paired, reused observations.

If float64 satisfies the implementation checks but the repair criteria still fail, report a valid negative result for the fixed first-block transfer test on the reused sample. If all criteria pass, report success only in this disclosed precision repeat; do not rewrite the original result or call it an independent confirmation. If state validity still fails, report an unresolved implementation or numerical discrepancy and do not use the repeat to validate the repair claim. Unexpected behavioral changes require investigation and disclosure, not a further unregistered search for favorable settings.

First-block state restoration is an algebraic implementation check, not a theorem that restores answers through GPT-2's remaining layers. Neither outcome establishes a universal mechanism in pretrained language models. This remains one fixed pretrained checkpoint, one selected lookup format and an externally specified position-and-mask intervention.
