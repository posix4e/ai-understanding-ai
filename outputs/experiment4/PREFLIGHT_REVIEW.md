# Experiment 4 prospective independent preflight

The input audit, intervention runner, numerical forecasts, scorer, reporting logic and registration/controller checks pass. This review clears the prospective freeze; it does not provide evidence of an E4 outcome. No E4 trained checkpoint was loaded or forwarded for this review; random-weight fixtures are explicitly separate from scientific outcomes.

## Fresh input verification

The reviewer independently enumerated all 24 pair orders and four query choices of each selected unordered association dictionary using scalar Python base-32 encoding. The historical E1 set was reconstructed through the already audited E2 helper; E2 recipient/donor arrays and E3 original/grouped inputs were then added using the independent encoding. This is independent selected-variant enumeration and metadata checking, not a wholly separate implementation of the historical generator.

All 196,608 variants have zero overlap with 2,853,613 prior inputs; grouped inputs also have zero overlap. The 2,048 unordered dictionaries are distinct, contain four distinct valid keys/values, and have consistent query tokens, class targets and value metadata. Each query index has exactly 512 rows. The selection audit records 2,114 candidates and 66 rejections; this bookkeeping is consistent, while rejection order was not replayed.

Input SHA-256: `9ee60746699fd35a084a714af9e01111d716a01f31da06ebd1556d4b63c681a0`. The forbidden-set count and hash also match the generator. Full evidence is in `preflight_data_verification.json`.

## Causal intervention and tests

The 15-condition grid includes the complete nine-mask family. Each mask deletes exactly six key edges per head, with per-value counts 3/2/1/0, and retains each value’s own key. The correct mask is `guard_a0_b3`; all eight alternatives are retained. V0’s mask is identical in every condition, and removed probability mass is not matched.

Static inspection confirms that non-oracle guards use canonical grouped Q/K only, recompute stable masked softmax for affected value rows, preserve physical causality, and copy every unaffected row. They do not use native activations or target classes to construct the patch. Native states enter only the explicitly named restoration oracles at the layer-2 input hook, with token-aligned indices. Unselected states remain endogenous to the current guard.

All ten random-weight runner tests and ten outcome-free synthetic scoring tests pass (20 total). An independent live-score oracle intercepts the actual root model’s first attention softmax and inserts the edge mask there, avoiding the runner’s cached Q/K computation; both the correct and a wrong guard agree. Tests cover underflow robustness, unchanged key/query rows, full-class oracle equivalence, actual layer-1 versus patched layer-2-input diagnostics, batching and metadata validation. The full native-state oracle and exact layer-1 recovery are algebraic checks, not empirical discoveries.

## Predictions and integrity gates

All 180 numerical forecasts were independently reconstructed from the declared formula and E3 reference means. The E3 summary source hash matches. The sham mixture assumes local query repair despite possible competition from other value rows; this is correctly described as an untested extrapolation rather than a consequence of the restoration theorem.

The manifest includes the E3 data helper, E3 input file and E3 summary used by the predictor, in addition to E4 code/data and all six checkpoints. Registration refuses existing manifests/locks and retrospective confirmation. Remote verification checks the manifest and every Git blob against local bytes. The controller checks local hashes and existence of the exact registered remote commit before its start record or loading a checkpoint, refuses an existing confirmation directory, runs all six seeds and retains partial failures.

The final scorer enforces the full-class guard/value-oracle and full-oracle/native distribution checks, alongside all 24 query-stratum accuracy cells and both strict paired confidence-bound gates per seed. It validates saved class probabilities, argmax, target probability, accuracy, residual errors and attention arrays. The same query-stratified row weights apply to all conditions and models; the eight-sham mean is computed within each row before resampling. Best-control selection is repeated in each bootstrap draw for its explicitly descriptive interval.

Synthetic tests detect altered non-target probability mass despite unchanged target probability, each residual-identity failure, native/no-op failures, a single failed query stratum, and recovery versus specificity failures separately. Equality fails both strict confidence thresholds; state error exactly at tolerance passes. The report recomputes the global conjunction and exposes every control, the best control, per-query outcomes, oracle checks and numeric misses. It separates guaranteed identities from empirical recovery and explicitly preserves E3’s failed result.

Current prior E1-v2 (45 files), E2 (62 files) and E3 (42 files) locks retain all hashes. There is no critical blocker to freezing the reviewed E4 implementation. All source/data/forecast bytes must still receive the planned remote verification before any trained forward.
