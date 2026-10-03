# E6: pretrained GPT-2 lookup calibration

This plan is fixed before any GPT-2 forward in this project. The checkpoint is
`openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
The original E1–E5 files and results remain unchanged.

## Purpose and scope

Test whether the original first-block visibility repair transfers to one
pretrained, 12-layer language model. GPT-2 small is approximately 124 million
parameters, compared with 70,720 in the custom model. No training or tuning of
weights is permitted. All inference runs locally on CPU, float32, four threads,
eager attention, dropout disabled, without a KV cache.

GPT-2 has established uses in causal circuit studies, including
[indirect object identification](https://arxiv.org/abs/2211.00593) and
[greater-than prediction](https://arxiv.org/abs/2305.00586). These are precedents
for studying pretrained GPT-2, not prior claims about this exact repair.

## Bounded native-only calibration

Generate 64 random four-pair dictionaries with seed 10600001, 16 queries per
pair position. The 16 key words, 16 numeric answers (20–35), and four completion
formats are fixed in `data.py`: table, repeat, one demonstration, two
demonstrations. The demonstrations use different labels and values.

Use all 64 cases in all four formats. Run only the original pair order, with
ordinary positions and no intervention. Do not inspect any reordered or repaired
GPT-2 outcomes during calibration. Select the format by highest 16-choice answer
accuracy, then highest raw correct-token probability, then declared format
order. Publish every result, including unsuccessful formats. No additional
format may be added after observing these scores in this experiment.

The primary score ranks all 16 permitted answer tokens. Also retain unrestricted
next-token predictions, exact next-token accuracy, raw target probability and
total answer-token probability mass. Restricted-choice accuracy must never be
reported as unrestricted generation accuracy.

Require at least 80% native 16-choice accuracy overall and 70% in each query
stratum for an adequate repair-test task. If calibration misses this criterion,
retain the best format and report the limitation; do not claim successful task
transfer. Fresh confirmation evaluates competence again without dropping cases.

## Token audit and subsequent confirmation

Tokenize the complete native prompt once. Verify that no tokenizer token crosses
the fixed prefix/key/value/query chunk boundaries, that key chunks have equal
length, and that each permitted answer is one token in the output context. Move
exact token occurrences with original position IDs; do not retokenize grouped
text. Prefix and query suffix remain fixed. The grouped text is a controlled
serialization intervention and need not be ordinary English.

After calibration, freeze the selected format, a fresh input set, conditions,
code, predictions and decision thresholds in a separate public registration
before any shifted or repaired GPT-2 inference. Plan for 512 fresh dictionaries;
use 256 only if measured native throughput predicts more than one hour for the
15-condition confirmation. Document that decision before freezing.

The 15 conditions will include native, grouped physical positions, grouped
canonical positions, a no-op, nine degree-matched value-row masks (correct plus
eight alternatives), and two separately labelled deeper-layer applications.
The first-block hypothesis is primary. Middle-block and all-block interventions
are secondary and cannot rescue failure of the primary claim.

First-block value/query restoration is an implementation identity under the
stated conditions. Final answer recovery through 12 layers is an empirical
hypothesis. Restoring first-block key states alone is not a valid full-output
identity control in this deeper architecture. This is one pretrained checkpoint,
not multiple independent model replications. We cannot audit absence from
GPT-2's unavailable pretraining corpus; inputs are newly generated combinations.
