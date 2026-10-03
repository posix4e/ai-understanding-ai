# E5 independent preflight review

Status: preflight passes; no remaining critical blocker found in the reviewed source. Nineteen combined random-weight and synthetic scorer/report tests pass. No E5 pretrained-model intervention was run or inspected in this review.

## Data and contamination

The independent input audit confirms 1,024 valid dictionaries, 256 queries per pair position, and a fixed 256-row boundary prefix with 64 per position. It reconstructed 256,000 training examples from each of the six new training streams and independently added the saved historical E2–E4 inputs to the previously audited E1 exclusion reconstruction. There are 4,393,029 forbidden exact input identities. All 98,304 native pair-order/query variants and all 10,321,920 legal physical-order/query variants of the selected dictionaries have zero overlap. Selected dictionaries have distinct association sets and token bags; none matches the 4,095 prior E3/E4 token bags. Targets, token ranges, uniqueness and query membership agree. The exact selection/rejection order was not independently replayed.

Input SHA-256: `75ca219d2b023d392f255c89a76f8951fb852e6c07c1f578394206dbf77756e4`. Full details and training metadata hashes are in `preflight_data_verification.json`.

## Structural and statistical checks

Independent enumeration and nine random-weight tests cover the 2,520 legal orders, 105 primary layouts, 2,415 boundary layouts, 729 matched guards (105 correct and 624 alternatives), 1,044 primary conditions, and three primary layouts lacking alternative masks. All 105 correct-mask restoration/oracle identities and all 2,415 own-key+self value-state identities were tested on synthetic inputs. An independent live attention-softmax intervention agrees with the cached-Q/K runner for primary, boundary and edge examples. Causality, row-degree matching, retention of each own key, unchanged key/query rows, stable masked softmax and batch schemas are tested. No native state enters a non-oracle guard; native caches serve diagnostics and explicitly labeled oracles only.

The scorer uses original-target probabilities from saved full 16-class probabilities, validates argmax/accuracy consistency, aggregates alternatives within layout and then the 102 eligible layouts equally, and aggregates recovery over all 105 layouts. Dictionary rows, not layouts, are bootstrap units. Shared query-stratified weights use 2,000 draws and registered seeds 10200001 and 10200002. The edge estimand correctly excludes the true answer and named wrong answer from the two control wrong values, conditions on the source-key query, and weights six edges equally. Every primary layout/query cell must meet .95, native-deficit upper bound must be <=.01, and specificity lower bound must be >.02 in every seed. Boundary and edge secondary decisions cannot rescue a primary failure.

An additional independent fixture assigns distinct effects to each edge and varies effects across dictionary rows. Direct equal-edge mean and query-restricted bootstrap calculations match the scorer exactly, including confidence bounds. Synthetic scorer tests cover nested control weighting, individual failed cells, strict and non-strict threshold boundaries, invalid arrays, native/no-op/state/oracle failures, missing panels, and report separation of failed primary and successful secondary claims.

The 72 forecasts contain the same 12 named aggregate endpoints for each fixed seed 6–11. Their .05 tolerance is separate from scientific gates. The prior E1v2, E2, E3 and E4 manifest dependencies were rehashed unchanged: 44, 61, 41 and 43 files respectively, excluding each manifest itself.

## Execution and mathematical audit

The model entrypoint verifies the exact public lock and every frozen local file before loading a checkpoint. It refuses to overwrite an existing or completed run. Explicit crash resumption rechecks the registration and checkpoint identities, requires saved shard SHA-256 values and layout IDs to agree, and rejects unindexed saved shards. The scorer requires the completed `finish.json` record, checks all saved shard hashes and panel coverage, and retains every condition/query result, every primary failed cell and every individual wrong primary answer. A completion-filename mismatch found during review was corrected before freeze. Registration refuses retrospective preparation and lock replacement; remote verification compares the public manifest and recursive Git blob identities with local bytes. Review did not perform public verification, which remains required before inference.

The prospective `THEORY.md` argument is valid for the fixed architecture: exact source-set restoration, the 105-layout characterization/count, final-query invariance, the missing-source counterexample, the layer-2 key-token-group mixture identity and the own-key+self reference distinction. These arguments do not imply correct task answers.

## Limits on interpretation

A pass would support transfer of an externally specified mask across the declared class and six new training seeds. It would not establish spontaneous generalization, an architecture-independent mechanism, a unique/minimal repair, or superiority over every matched alternative. Matching key counts does not match removed attention mass. The full oracle and value-state identities are architectural checks, not learned discoveries. Own-key+self supplies explicit association structure. The 23-layout fixed-key-order boundary subset plus grouped primary reference separates available-key context from reordered values; the remaining boundary class varies both missing keys and values.

All-layout observed accuracy gates are not simultaneous confidence guarantees for population accuracies. Bootstrap intervals condition on these checkpoints; six models and shared dictionaries are not thousands of independent replications. Procedural predictor blinding occurs in a shared filesystem. E3's failed positional-sufficiency prediction and all E5 misses must remain visible.
