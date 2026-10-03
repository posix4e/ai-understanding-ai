# E6: frozen transfer test in pretrained GPT-2 small

## Question

Does the first-block visibility repair from E4/E5 restore lookup behavior after
input reordering in pretrained GPT-2 small? This is one unchanged pretrained
checkpoint, not a test that isolates model size from architecture or training.
The model has 124,439,808 learned parameters and 12 transformer blocks. The
custom E5 model had 70,720 parameters and two blocks. Inference stays local:
CPU float32 model computation, four threads, evaluation mode, eager attention,
no KV cache. Normalize final logits into probabilities using float64 softmax.

## Calibration and fresh data

The native-only calibration was publicly fixed in commit
`38334a0670adc6980a802d39621e64f5d27cd1c0` before any GPT-2 forward here.
All four formats and all 64 calibration examples are retained. The declared
selection chose `one_demo`. Its native 16-choice accuracy was 62/64 and its
unrestricted next-token accuracy was 59/64. No reordered or masked pretrained
inputs were tested during selection. Those observations are setup results.

Confirmation uses 512 newly generated four-pair dictionaries (seed 10600002),
128 queries per pair position. Reject duplicate association maps and every
calibration association map, regardless of pair order or query. The 16 key words
and 16 answer values remain fixed. The same 512 cases appear in all conditions.
We cannot establish absence from GPT-2's unavailable pretraining corpus.

## Exact input and mask intervention

The selected native prompt contains one fixed completed demonstration, a new
table, and its query. Tokenize the complete native prompt once. Verify exact
roundtrip, unsplit chunk boundaries and single-token answers in the actual
output context. Each of four key chunks has two tokens; each value chunk also
has two. Preserve the prefix and query suffix. Group all four key chunks before
the four value chunks, without retokenization. Original position IDs travel
with each token in canonical-position conditions. The grouped text is an
artificial serialization intervention, not a normal-language paraphrase.

At each value chunk Vi, the correct guard removes all token edges from future
logical key chunks Kj, j>i. It retains the value's own key, all physically
available value tokens, fixed prefix and self. Apply it in every attention head
of block 0 only for the primary hypothesis. Every mask includes physical
causality, and no source is added. No activation transplant or parameter change
is used. The experiment supplies the true pairs and original position tags.

## Fifteen conditions, fixed before confirmation

1. Native order and positions.
2. Grouped order with physical positions.
3. Grouped order with original positions.
4. Condition 3 with the unchanged physical causal mask explicitly reapplied.
5. Correct guard in block 0.
6–13. All eight other matched guards. For V0–V3 retain respectively 1, 2, 3,
   and 4 key chunks, always including the own key. These controls match token
   edge counts but not removed attention mass or perturbation size.
14. Correct guard in block 5 only (secondary).
15. Correct guard in all 12 blocks (secondary).

The last two conditions cannot rescue a failed primary hypothesis. Without
their own matched controls, their improvements would not establish specificity.
Report every condition and every query stratum, including the best alternative.

## Implementation validity

Before trained confirmation, random-weight tests must verify stock/no-op
agreement, physical causality, layer scope, mask row counts and exact predecessor
sets. Recheck these identities in the actual model: full-vocabulary no-op
probability error ≤1e-6; native versus correctly guarded first-block value-state
error ≤1e-5; first-block query-suffix invariance ≤1e-5; and unedited first-block
key states versus unguarded grouping ≤1e-5. Require complete finite outputs and
unchanged parameter hashes. These are implementation checks, not discoveries.

GPT-2's later blocks can change restored value states again. First-block
restoration alone therefore gives no theorem about final answers. In particular,
do not apply E5's two-block full-output oracle identity to this 12-block model.

## Scores and fixed decision rules

Primary accuracy chooses the highest-probability answer among all 16 numeric
answer tokens, not only the four displayed values. For specificity, normalize
over those 16 tokens and compare correct-answer probabilities. Also publish
unrestricted next-token accuracy, raw target probability, total 16-token
probability mass and decoded unrestricted predictions. Never describe the
restricted score as ordinary generation accuracy.

All implementation checks and all five behavioral criteria must pass:

| Criterion | Fixed requirement |
| --- | --- |
| Native task competence | 16-choice accuracy ≥.80 overall and ≥.70 in each of four query strata |
| Ordering damage | Native minus canonical-grouped accuracy ≥.10, with paired 95% interval lower bound >0 |
| Repair benefit | Correct-guard minus canonical-grouped accuracy: paired 95% lower bound >.05 |
| Near-native recovery | Native minus correct-guard accuracy: paired 95% upper bound ≤.05 |
| Specificity | Correct-guard minus mean-eight-shams conditional target probability: paired 95% lower bound >.02 |

Use 2,000 paired query-stratified dictionary bootstrap draws, seed 10600003,
sharing weights across conditions. Average the eight shams equally within each
dictionary before resampling. Dictionaries are the resampling units; conditions
and token edges are not independent observations. Intervals apply to this fixed
checkpoint and sampling design. Six blind numerical forecasts are assessed
separately with absolute-error tolerance .10; they are not causal pass criteria.

Classify outcomes in order: implementation invalid; native task inadequate;
ordering failure not reproduced; primary repair criteria failed; all primary
criteria passed. Publish all metrics even when an earlier gate fails. Secondary
results and individual examples remain descriptive.

## Registration and reporting

Freeze this protocol, code, exact conditions, input cases, tokenizer/model file
hashes, dependencies and predictions in a public commit before any reordered or
repaired GPT-2 forward. Verify local bytes against that commit and the model
download hashes. Save all raw candidate probabilities and full-vocabulary
predictions by condition, plus batch hashes and state-check diagnostics.

Do not change thresholds, select cases, swap the model, add prompts, search
layers, retrain, or remove failures after observing confirmation. A later study
requires a new labelled plan. A pass would establish one bounded transfer of a
supplied repair. A failure would locate a limit of that repair. Neither outcome
establishes a universal mechanism or shows that scale alone caused the result.
