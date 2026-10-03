# Experiment 3: answer-specific binding through positional coordinates

## Question and scope

Does changing only the assignment of already-trained position embeddings make
the model retrieve the particular value implied by those coordinates? E1 located
key-dependent routing at value positions. E2 predicted response to component
corruption using a fitted statistical surrogate. E3 asks a different, more
structural question without any E3 discovery measurements or per-model fitting.

The six existing 70,720-parameter checkpoints are reused unchanged. No new
training, cloud compute or inference APIs are used. All actual transformer
execution is local CPU. The conversational hypothesis/review agents are not
claimed to be locally hosted models. Prediction blinding is procedural on a
shared filesystem, not enforced access isolation.

The fixed grid tests supplied positional coordinates. It does not test natural
understanding of a new serialization, recover a unique internal algorithm, or
establish that this phenomenon is unprecedented in the literature.

## Manipulations and specified alternate answers

Each example contains four distinct associations `(Ki, Vi)` and query `Kq`.
The original sequence is `K0 V0 K1 V1 K2 V2 K3 V3 Q`. The grouped sequence is
`K0 K1 K2 K3 V0 V1 V2 V3 Q`. All conditions retain the same tokens, nine-token
length, query at physical index 8, target semantics and trained weights.

Let `pi` be a permutation of the four slots:

* Coherent mapping: assign Ki position ID `2*pi[i]`, Vi ID `2*pi[i]+1`.
  The predicted answer remains Vq.
* Value-only mapping: assign Ki ID `2*i`, Vi ID `2*pi[i]+1`.
  The predicted answer is `V[inverse_pi(q)]`, which can differ from Vq.

Both mappings use every trained ID 0 through 8 exactly once. Key IDs stay even,
value IDs stay odd, and query ID stays 8. This controls key/value parity and the
multiset of positional vectors. In the grouped layout every candidate key is
causally available to every value. Physical causal masking remains unchanged;
it is never reordered by the supplied IDs. The input residual is replaced using
the existing layer-0 `resid_pre` hook by token embedding plus the assigned
position embedding. This intervention changes no parameter.

All 24 permutations are included, with the identity shared as
`grouped_canonical`. There are 23 coherent and 23 value-only permutations, plus
original-native, original-no-op, grouped-native and grouped-canonical: 50
conditions per model. Grouped-native uses ordinary physical position IDs; its
"slot" endpoint is a duplicated original-target reference, explicitly flagged
as lacking a defined slot map. Nine permutations have no fixed slots
(derangements); those and their nine coherent controls are primary. The other
14 nonidentity permutations and native-layout rescue are secondary.

## Inputs and pre-outcome safeguards

Freeze 2,048 new association dictionaries from generator seed 7100001. Query
index is the accepted row index modulo four, giving exactly 512 per stratum.
All 96 serializations of a candidate dictionary (24 pair orders times four
query choices) are checked against the reconstructed forbidden set covering
audited E1/E2 training, validation, discovery and confirmation inputs. Reject a
dictionary if any variant overlaps. Reject unordered association duplicates
within E3. Outcomes never enter selection. The input audit retains all rejects.

Every condition and model receives the same dictionaries. No new trained-model
forward occurs before public registration. Synthetic runner tests use freshly
initialized random models only. Freeze inputs, checkpoints, source, forecasts,
thresholds and independent preflight review. Push the registration commit and
read back the exact remote manifest and recursive Git tree, checking every blob
against its local content, before creating the confirmation start record.

`run_phase.py` rechecks local hashes and existence of that exact remote commit,
then refuses overwrite/restart if confirmation already exists. A failed runtime
is retained; repairs require explicit disclosure and a new registration if they
affect the frozen analysis. A scientific failure is a result, not a trigger to
change gates or search for a favorable seed.

## Endpoints, uncertainty and decision

Save per-case original-target probability, specified slot-target probability,
their argmax accuracies, actual argmax class, and four-head attention diagnostics
for all 300 model-condition cells. Attention diagnostics summarize L1
value-to-key and L2 query-to-value weights. Every head is retained and equally
weighted for summaries; these are descriptive observations, not mediation tests.

The mechanistic conjunction requires, for **each of six models**:

1. Original-native accuracy at least .95, finite arrays, and original no-op
   maximum target-probability discrepancy at most 1e-6.
2. Specified alternate-answer accuracy at least .90 for **each** of nine value
   derangements, and original-answer accuracy at least .95 for **each** matched
   coherent mapping.
3. The lower paired 95% bootstrap bound on mean
   `P(slot answer) - P(original answer)`, averaged equally over the nine
   derangements, strictly above .80.

The third condition concerns confidence, not just argmax. Native-grouped failure
is not required. Report its change under canonical IDs separately. Do not claim
all permutations passed when only primary ones do. Record every failed cell.

Use 2,000 multinomial bootstrap replicates, seed 7200001, stratified within the
four query groups. Resample whole dictionary rows with identical weights across
conditions and models. Average the nine paired contrasts per row before
resampling. Report per-condition Wilson 95% intervals for accuracy and paired
intervals for the main contrasts. Individual intervals are not simultaneous
confidence bands. All uncertainty conditions on these six checkpoints and this
sampling design; six models are not thousands of independent training replicas.

The blinded predictor commits 1,200 numerical forecasts (50 conditions times
six models times four endpoints) and an answer-identity rule. Report every
absolute error, RMSE, and count within .15, separately from the mechanistic
conjunction. Numeric forecast failures cannot be hidden by passing causal gates.

Answer-fidelity comparators predict the observed model's answer: original
semantic answer, exchangeable four displayed values, and uniform 16 classes.
A structural competing rule uses `V[pi(q)]` instead of `V[inverse_pi(q)]`.
Compare those directions only on the 14 noninvolutive permutations and query
indices where their named values differ. It is a secondary check, not a
post-hoc new success gate. These comparators are hypothesis controls; no claim
is made that they exhaust empirical prediction algorithms.

## Interpretation and relation to prior work

Specific wrong-answer selection with coherent restoration would support causal
binding through the supplied position codes. Rescue alone is weaker: it could
just restore key/value role parity. Even the stronger result does not distinguish
an absolute-slot table from a learned previous-coordinate computation using
absolute embeddings. Physical context still changes intermediate attention
competition, and attention correlations do not locate a sufficient circuit.

[Singh et al., July 2026](https://arxiv.org/html/2607.18759v1) study positional
pinning and length transfer, prominently involving out-of-range untrained
coordinates. Here all coordinates were trained, are in range, and occur once
in every remapped example. [Ruoss et al., 2023](https://arxiv.org/abs/2305.16843)
study randomized positional encodings during training. This experiment instead
tests fixed models through inference-time coordinate reassignment. These scope
differences motivate the test; they do not prove a field-first contribution.

A closer precedent is [Tang, Lake and Jazayeri, February 2026, Fig. 12](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340088):
targeted positional swaps in an internal retrieval pathway redirect attention
and partially alter output logits. Therefore positional intervention causing
answer-specific redirection is already known. Our complete slot-permutation
grid, parity-preserving matched controls and six-seed prospective quantification
test a narrower reliability claim. See `outputs/experiment3/RELATED_WORK.md`.

Run the complete grid once and publish raw arrays, all misses, independent
review and executable reproduction instructions. Any follow-up requires a
separate prospective protocol. Do not repeatedly resample until significance.
