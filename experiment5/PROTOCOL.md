# E5: prospective transfer of causal-visibility repair

## Contribution and cohort

E3's position-only explanation failed. E4's value-row key-visibility repair
succeeded in one grouped layout across six already studied models. E5 tests
the unchanged rule on a complete finite serialization class and six newly
trained models. The claim concerns externally supplied structure, not spontaneous
layout understanding, a unique algorithm, or a new general binding principle.

Seeds 6–11 and the unchanged 2,000-step native-layout training recipe were
published in commit `d85ac669c219ca15c0fd6752efac797dd2eb206a` before training.
Every final checkpoint is retained; no seed replacement or training extension
is allowed. Training validation is disclosed monitoring, not confirmation.
The predictor may use E1–E4 results but no E5 inputs, checkpoint metrics or
model outcomes. Blinding is procedural in a shared filesystem.

## Structural proposition and empirical question

Canonical token identities are K0,V0,K1,V1,K2,V2,K3,V3,Q, numbered 0–8. Their
learned position IDs travel with them; the query remains last. Of 2,520 physical
orders placing every Ki before Vi, exactly 105 preserve value order. For every
Vi in those 105, all native predecessors are still available. Deleting visible
Kj with j>i leaves exactly its native predecessor set. The first-layer value
and query states therefore equal native states up to floating-point ordering.
Key states remain altered and can affect the second layer: final-answer
recovery is an empirical prediction, not this identity.

If value order changes, at least one value lacks a native predecessor; deletion
alone has no weight-independent restoration guarantee. This does not imply a
trained model needs every missing input. The boundary tests distinguish an
architectural guarantee from actual behavior. A proof and counterexample will
be included in the manuscript; the proposition is elementary and is not being
claimed as a novel general theorem.

## Complete primary grid

Test all 105 layouts, including native and canonical grouping as labeled
references. At Vi, enumerate every subset of currently visible keys of size
i+1 that retains Ki; retain all physically available values. Across layouts
this produces 729 guards: 105 correct masks and 624 alternatives. Three layouts
have no legitimate matched alternative. For all 105 include unguarded, no-op
attention reapplication and correct guard plus native-key restoration at L2
input. Total: 1,044 primary layout/condition cells per model.

All masks use current-layout Q/K and stable masked softmax. Only L1 value rows
are edited. Native activation transplantation occurs only in the explicit
oracle. The mask uses the known pair mapping; it never consults target answers.
Controls match row degrees and retain true keys, not removed attention mass or
activation norms. The oracle is a guaranteed positive control, not a discovery.

## Boundary and edge panels

Evaluate all remaining 2,415 layouts under four policies: unguarded; E4 keyguard
deleting future-logical keys; logical-prefix intersection deleting all visible
tokens with native ID later than Vi; and own-key+self keeping only Ki and Vi.
The last policy imposes the strongest explicit pairing prior. Its L1 value
states equal an identically masked native reference under every layout by
construction; final accuracy remains empirical. Missing prior keys and values
both vary in this complete boundary panel. Also report the fixed-key-order
subpanel K0 K1 K2 K3 followed by all 24 value orders (23 boundary plus grouped
primary), which holds all keys available while previous-value context varies.

From the correctly guarded grouped layout, separately re-add each of six edges
Vi←Kj for i<j. On cases querying Kj, predict increased probability of the
specific wrong answer Vi. The endpoint is the change from the correct guard
in [P(Vi) minus the mean probability of the other two displayed wrong values].
Average the six eligible-query edge means equally. Query weights are therefore
1/6,2/6,3/6 for j=1,2,3; q=0 is ineligible. Save all query outcomes regardless.
This tests competition from another value while the queried value stays intact.

## Data, preregistration and execution

Use 1,024 new association dictionaries, exactly 256 per query position, shared
across models and primary/edge conditions. Boundary uses the first 256 accepted
rows, exactly 64 per query. It is a fixed subset, not independent confirmation.
Exclude all 96 pair-order/query variants against all audited historical inputs
and all fresh-model training data. Additionally reject prior E3/E4 key/value
token bags regardless of pairing, preventing reordered grouped-input overlap.

Freeze code, conditions, inputs, forecasts, checkpoints, protocol and independent
preflight checks in a public commit and verify every Git blob before any E5
trained-model forward. All computation stays on local CPU. Save full 16-class
probabilities, predicted classes and state discrepancies for every condition,
in fixed 32-layout shards; no activation caches are published. A start record
and completed shard ledger make partial runs visible. A crash may resume only
after re-verifying frozen identities and previously saved shard hashes, without
overwriting outcomes or changing scientific decisions.

## Gates and statistics

Every seed must pass all validity checks: native accuracy ≥.95; finite, complete
and consistent arrays; full-class no-op error ≤1e-6; correct-guard L1 value/query
restoration and unchanged key rows ≤1e-5; full oracle/native probability error
≤1e-5. Boundary own-key+self value-state identity is checked against its native
reference within 1e-5. Query invariance and unedited key rows apply to all masks.

Primary recovery requires observed accuracy ≥.95 in each of all 105×4
layout/query cells per seed. The paired 95% upper bound on the native-minus-guard
mean target-probability deficit must be ≤.01, equally averaging 105 layouts.
Specificity requires the paired 95% lower bound on guard-minus-mean-sham
probability >.02. First average all alternatives within each layout, then
equally average the 102 eligible layouts. Report best alternatives descriptively;
an average advantage is not uniqueness or superiority to every alternative.

Use 2,000 query-stratified dictionary-row multinomial bootstrap draws, seed
10200001 for primary/edge and 10200002 for boundary. Share weights across layouts,
conditions and models. Aggregate within dictionary before bootstrapping. Edge
weights are normalized within the appropriate source-key query stratum before
equally averaging edges. Intervals condition on these checkpoints and sampled
dictionaries; per-cell Wilson intervals are not simultaneous confidence bands.
Layouts and shared dictionaries are not independent model replications.

Separate secondary predictions: logical-prefix and own-key+self boundary
accuracy averaged over all 2,415 layouts ≥.95 per seed; edge competitor-enrichment
95% lower bound >.05 per seed. Report the fraction of boundary layout/query
cells meeting .95, without using that fraction as an extra gate. None of these
secondary decisions can rescue a failed primary conjunction.

The predictor freezes 72 aggregate numerical forecasts, 12 per model, with
absolute-error tolerance .05. These are not per-condition forecasts and their
coverage is distinct from the causal gates. Retain all forecast misses, primary
failure rows, individual controls and boundary outcomes. No failed model, layout
or query can be dropped, and no thresholds may change after observation.

## Interpretation and publication

A pass supports transfer of a specified partial repair across new models and
the complete stated class. Failure identifies a limit even if pooled accuracy
is high. Broader boundary success would show that exact prefix reconstruction
is stronger than necessary; it would not retrospectively extend the theorem.
Prior binding, positional-steering, mask-graph and context-restoration work
constrains the novelty claim; see the public publication gap review. The
manuscript must preserve E3's failure and every E5 counterexample.
