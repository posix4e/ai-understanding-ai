# Experiment 5 prospective skeptical design review

E5 addresses the largest publication gap in E4: whether its fixed repair transfers without fitting to new checkpoints and an exhaustively specified serialization class. It must retain E3's failed explanation and E4's successful but non-unique approximate repair. This review contains no new model outcomes.

## Primary contribution and exact universe

The primary universe consists of all 105 labeled key/value orders satisfying Ki before Vi and `V0<V1<V2<V3`, with the final query last and canonical position IDs traveling with tokens. It is not all 2,520 key-before-own-value orders; the remaining 2,415 form a separate boundary panel. Requiring key order as well would reduce the primary universe to 14 and weaken the proposed test.

In each primary layout, every native predecessor of Vi is present. Deleting visible Kj→Vi edges with j>i restores its complete first-layer predecessor set. Pointwise operations and source-order symmetry then give native value outputs; the final query's first-layer output is also invariant. Key states need not be native. This proposition establishes a manipulation guarantee, while near-native final answers remain a falsifiable empirical claim.

Enumerate every row-degree-matched key mask preserving the correct-key edge at each value. Their 729 total layout/mask combinations include the correct guard for each layout; 102 layouts have genuine alternatives. Baseline, no-op and guard-plus-native-key oracle at each layout give the declared 1,044 primary conditions. The oracle restores the whole L2 input to a permutation of native states and therefore is a guaranteed output check, not independent evidence for repair. The native and previously studied grouped layouts must remain labeled references in layout-transfer summaries.

The complete family is preferable to a random convenient sham. Average alternative masks within layout, then average the 102 eligible layouts equally. Pooling all alternative conditions directly would overweight layouts with many masks. Layouts with no alternative must not receive an invented zero difference or duplicate guard as a control. They remain part of the accuracy and native-deficit analyses.

## Falsifiable primary conjunction

Use all six independently trained final checkpoints, seeds 6–11, from the publicly fixed recipe. Do not replace a failed seed, choose an earlier checkpoint, extend training, fit guard parameters or calibrate confidence forecasts to any E5 shifted outcome. The architecture and training distribution remain fixed, so this tests transfer across training randomness and serialization, not across architecture families.

On the same 1,024 fresh dictionaries, exactly 256 queries per pair index, require in every seed:

* Native accuracy >=.95 and complete finite outputs, with declared no-op and architectural identity checks.
* Correct-guard accuracy >=.95 in every primary layout/query cell: 105 × four cells per checkpoint, without averaging away an unsuccessful layout.
* Paired 95% upper confidence bound for native-minus-guard original-target probability, averaged equally over 105 layouts within each dictionary, <=.01.
* Paired 95% lower confidence bound for correct-guard probability minus the mean available alternative-mask probability, averaged equally over the 102 eligible layouts within each dictionary, >.02.

All six seeds must pass for the uniform primary claim. Resample complete dictionary rows within query strata, sharing weights across layouts, masks and models. Do not treat hundreds of layouts as independent new datasets. Individual Wilson intervals are not simultaneous population guarantees; distinguish observed cell thresholds from confidence statements.

If any cell fails, the uniform transfer claim fails even if pooled accuracy is high. If only the full native-state oracle succeeds, the guarantee holds but the partial repair is insufficient. If specificity fails, generic sparsification remains competitive under these controls. Such failures may identify a valuable boundary, but must change the manuscript claim rather than the frozen gates.

## Boundary panel: useful but logically separate

Evaluate all 2,415 remaining key-before-own-value layouts on the first 256 cases of the frozen input array, retaining exact query balance. Fix that prefix before outcomes. The four policies are unmasked canonical coordinates, the same key guard, logical-prefix restriction, and own-key-plus-self attention at value nodes. None may add physically future attention edges.

The key guard deletes only newly exposed future-logical keys. A logical-prefix mask can also remove future-logical value nodes, but cannot restore missing native predecessors. Own-key-plus-self imposes a stronger explicit pairing prior than matching the native context; it must not be presented as an internal algorithm newly inferred from the trained network. Freeze these policies' exact query/source axes and their handling of self/query tokens.

Loss of the primary equality theorem does not imply behavioral failure. Conversely, a good pooled boundary result does not establish uniformly good performance across 2,415 layouts. The registered separate secondary criteria require pooled accuracy >=.95 for logical-prefix and own-key+self in every seed. These are distinct policy claims. The fixed-key-order subpanel keeps K0,K1,K2,K3 first and varies all 24 value orders: 23 boundary layouts plus the grouped primary reference. It is not the entire all-keys-first family with arbitrary key order. Report each policy, layout distribution, low-performing tail and worst cases. Preserve the denominator and never select a successful policy after results to rescue a failed primary claim.

## Edge panel: make the named error test precise

The six interventions restore one deleted Kj→Vi edge, j>i, to the correctly guarded grouped layout. The directional prediction is that querying Kj can retrieve Vi, the value attached to the newly contaminated node. It is not simply that any answer becomes less accurate.

The registered enrichment endpoint for each eligible query is the change, relative to the correct guard, in `P(Vi) - mean(P(other two wrong displayed values))`. This separates the specified competing answer from generic damage. A raw increase in P(Vi) alone is weaker evidence. The original answer is Vj and must be excluded from the two wrong-value controls. The registered secondary criterion is a lower 95% bound greater than .05 in every seed.

Average within each edge's eligible q=j cases, then equally across six edges if claiming an edge-balanced effect. Under that choice, q=1,2,3 contribute weights 1/6,2/6,3/6; q=0 is structurally ineligible. Averaging eligible edges within each row and then equally over query strata is a different estimand. Use 2,000 query-stratified shared bootstrap draws, seed 10200001 for primary and edge panels and 10200002 for the 256-row boundary panel. Normalize the primary weights within each edge's eligible query stratum before taking the equal six-edge mean. A >.05 lower-bound criterion in every seed is a demanding, separate secondary gate and cannot compensate for primary failure.

One-edge reinsertion tests sensitivity conditional on the other five edges remaining blocked. It does not prove all six deletions are necessary together, that the mask is minimal, or that effects add. Report every edge and model even if only the pooled enrichment is gated.

## Feasibility, blinding and retained evidence

The declared panels contain approximately 21.29 million model/case/condition evaluations across six checkpoints: 6.41 million primary, 14.84 million boundary and 36,864 edge evaluations, before shared references. This is finite local-CPU work with nine-token models. Avoid caching all Python-list/full-activation results in memory; use bounded batches and per-panel/per-seed saved arrays. Benchmark only random weights before the final freeze, and do not use speed pressure to drop unfavorable layouts later.

Independent input auditing must cover all new training batches and prior E1–E4 dictionaries, not just prior exact queried sequences. Preserve all 96 association-equivalent pair-order/query variants in the exclusion criterion; every E5 policy uses a known serialization of those frozen dictionaries. Reject prior E3/E4 unordered token bags regardless of association, preventing nonnative serialization overlap. Independently check all 2,520 legal orders and four query choices, not only the 96 native variants. Reconstruct targets and query balance independently. Native training validation is disclosed monitoring; it is not independent confirmation evidence.

Before new shifted forwards, freeze every policy, condition enumeration, source file, checkpoint, data byte, forecast, score, bootstrap convention and failure rule, with remote readback. Synthetic tests should verify the 105/2,415 partition, 729 masks, changed physical indices, deletion-only causality, endogenous unpatched nodes and theorem/oracle equalities. The frozen full-class no-op tolerance is 1e-6, and state/full-oracle identity tolerance is 1e-5. Query invariance and unchanged L1 key rows are checked across every panel and mask. These checks remain distinct from target-probability success metrics. The 72 numerical forecasts cover 12 aggregate endpoints per seed with absolute tolerance .05; they do not claim numerical prediction of every condition.

The publication contribution, if supported, is prospective transfer of a training-context repair across a complete finite class, with an explicit theoretical boundary and informative failures. It remains assisted intervention in small fixed-format models. Positional binding, context restoration and attention-mask effects have relevant precedents; success here does not establish a universal mechanism or field-wide priority.
