# Experiment 3 independent prospective preflight

The selected input data and the position-intervention runner pass this review. No trained checkpoint was loaded or evaluated. The final phase controller, scorer and aligned prediction/protocol prose also pass static review. This is clearance for the prospective freeze, not evidence of any Experiment 3 outcome.

## Input-only audit

Independently enumerated each selected dictionary using sorted key–value pairs, all 24 pair orders and all four query keys, with Python integer base-32 encoding independent of the selection helper. Rebuilt the earlier forbidden set using the previously audited E2 helper and independently added the E2 arrays. The independently calculated forbidden count and sorted-set hash match the generator record.

* 2,048 distinct unordered association dictionaries; four distinct valid keys and values each.
* Exactly 512 queries at each pair index. Query tokens, class targets and value metadata are consistent.
* All 196,608 dictionary/order/query variants have zero overlap with 2,849,517 prior training, validation, smoke, E1 and E2 inputs. The grouped sequences also have zero overlap.
* Generator bookkeeping reports 2,127 candidates and 79 rejected dictionaries; their sum matches the accepted count. The reviewer audited accepted membership and invariants, not replayed candidate rejection order.
* Input SHA-256: `94ab4afe95acc166945135aa306af3d9e8b08c280bef1de9d03d246f1ba1e041`.

Machine-readable evidence is in `preflight_data_verification.json`. The earlier-data reconstruction reuses an already audited helper, so the independence is specifically in selected variant enumeration, encoding, metadata checks and hash reconstruction, not a second implementation of the entire historical random-data generator.

## Intervention and target audit

The 50 conditions contain exactly 23 coherent and 23 value-only nonidentity maps, the collapsed identity map, and three reference/no-op conditions. Each family has nine derangements. Every edited map uses IDs 0–8 exactly once, preserves query ID 8, and leaves all hypothesized keys causally available before grouped values.

Static inspection confirms that the input patch replaces the first layer's pre-attention residual with the token embedding plus the assigned position embedding. Physical order controls the existing causal mask. It does not mutate embeddings or checkpoints. Coherent targets remain the original value. Value-only targets are `inverse(pi)[query_pair]`, while the hypothesized key for value i is `pi[i]`; these are different directions and are implemented correctly.

All eight synthetic runner tests pass with unrelated random initialization, and all eight synthetic scoring tests pass (16 total). The tests include an independently hardcoded noninvolutive map (`2013`), explicit loops for all-head attention diagnostics, exact no-op equality of stored outputs, condition enumeration, malformed metadata rejection and batching consistency. This validates tensor indexing and intervention assembly without testing the scientific hypothesis on a trained model.

`grouped_native` deliberately has no slot mapping. Its duplicate original-target slot metrics are reference fields and must remain labeled undefined in mechanistic interpretation. Attention summaries report all heads; L1 averages the four value positions, and L2 uses the final query. Attention is corroborative, not a causal mediation test.

## Claim limits and final freeze checklist

The primary claim requires every model and every derangement accuracy threshold, each matched coherent threshold, and the paired probability-gap confidence gate. It must not be replaced by the average across favorable conditions. Resample dictionary rows within query strata, sharing row weights across permutations and models; the nine maps are repeated measures of each dictionary. Individual Wilson intervals are not simultaneous intervals.

The outcome can establish answer-specific control by externally assigned in-range positional representations. It cannot establish spontaneous layout generalization, unique internal semantics, an absolute lookup table versus a preceding-coordinate operation, or priority over all prior literature. All six checkpoints are reused; no population-level model generalization follows from case-bootstrap intervals.

PREDICTION.md and protocol.json now agree on bootstrap seed 7200001 and the probability-based no-op gate. Full 16-class probabilities are retained, validated against saved argmax/target probabilities, and used for the prespecified forward-versus-inverse and matched coherent wrong-target comparisons. The scorer averages each dictionary’s nine primary contrasts before applying the shared query-stratified weights. Synthetic fixtures show that one bad derangement, one bad coherent control, invalid native accuracy, or excessive no-op error fails its respective gate; a contrast exactly .80 fails the strict lower-bound gate. All 1,200 forecasts were independently reconstructed from the written formula.

The controller verifies frozen local hashes and the registered remote commit before creating the start record or loading a checkpoint, refuses an existing confirmation directory, and preserves partial failures. Registration verifies the remote manifest and every Git blob, and now refuses replacing an existing manifest or registration lock. Current E1-v2’s 45 frozen files and E2’s 62 frozen files retain their hashes. The original aborted E1-v1 source differences belong to the already disclosed v2 amendment. No trained forward should occur until the final manifest, checkpoints, exact inputs, conditions, forecasts, scorer and controller have been remotely verified.
