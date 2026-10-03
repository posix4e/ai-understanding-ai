# Mathematical notes: causal visibility, partial restoration, and residual key routes

**Prepared 2026-10-03 before E5 confirmation.** These notes use the fixed architecture and outcome-independent condition construction. They use no E5 checkpoint metrics, evaluation inputs, or model outcomes. The arguments are elementary consequences of attention's symmetry over its source tokens and pointwise processing. They are not presented as novel mathematical theorems or as empirical findings.

## Architecture and notation

There are four distinct key tokens K0,…,K3, four distinct value tokens V0,…,V3, and a final query Q. The query repeats one key's token identity but is a distinct sequence occurrence. The native sequence is

\[
K_0,V_0,K_1,V_1,K_2,V_2,K_3,V_3,Q.
\]

Canonical position coordinates are c(Ki)=2i, c(Vi)=2i+1, and c(Q)=8. Every layout moves each occurrence together with its coordinate. Thus its input residual is always

\[
x_t=E(\operatorname{token}(t))+P(c(t)),
\]

independently of its physical location. This requires the explicit coordinate intervention: the ordinary model forward uses physical position IDs. Q remains physically last. A causal attention row includes its destination itself. Token occurrences, rather than token vocabulary identities, index predecessor sets.

The architecture has two pre-normalized transformer blocks. LayerNorm and the feed-forward network act separately at each sequence position. Attention is the only operation that mixes positions; there is no additional relative-position bias, convolution, cross-position normalization, or stochastic dropout. For a first-layer attention head, write

\[
q_t=W_Q\operatorname{LN}(x_t)+b_Q,\quad
k_s=W_K\operatorname{LN}(x_s)+b_K,\quad
v_s=W_V\operatorname{LN}(x_s)+b_V.
\]

For a nonempty allowed source set S at t,

\[
z_t(S)=\frac{\sum_{s\in S}\exp(q_t^\top k_s/\sqrt{d_h})v_s}
 {\sum_{s\in S}\exp(q_t^\top k_s/\sqrt{d_h})}.
\]

Multi-head concatenation and output projection produce the attention residual update. The subsequent residual addition, LayerNorm, MLP, and second residual addition are pointwise. The same arguments accommodate the affine biases in the actual implementation.

Let N(t) be the native causal predecessor set of occurrence t, including t. Let C_L(t) be its physical causal predecessor set in layout L. A deletion mask may choose a nonempty subset of C_L(t), with a stable softmax recomputed over that subset. It does not add a source, transplant an activation, or change the source embeddings.

## Proposition 1: exact first-layer restoration by deleting extra predecessors

**Statement.** If N(t)⊆C_L(t), retaining exactly N(t) in every first-layer head at t reproduces the native first-layer output state of t for arbitrary parameter values, in exact arithmetic.

**Proof.** Canonical coordinates preserve x_t and every source x_s. Consequently q_t, k_s, and v_s agree with their native counterparts. The intervention gives exactly the same terms in the numerator and denominator of z_t as native attention. Their physical enumeration order is immaterial in exact arithmetic. Every head output and the output projection agree. The destination residual is unchanged, and the remaining operations in this block are pointwise functions of the same vector. The complete first-layer output therefore agrees. □

The proposition applies to one destination at a time. First-layer sources are projections of input residuals, not of other destinations' updated first-layer states. It therefore does not require simultaneous restoration of all other destinations. Floating-point summation order can introduce small numerical discrepancies; the protocol's tolerances test implementation agreement rather than replacing the exact-arithmetic statement.

### Missing predecessors remove the universal guarantee

If some occurrence u∈N(t) is absent from C_L(t), there need not be any deletion-only restoration of t. The claim is existential over weights: it is **not** a prediction that every trained checkpoint must fail on every layout with a missing predecessor.

A counterexample can be constructed within the fixed architecture. Set all token embeddings to zero. Choose canonical position embeddings so that every occurrence other than u has input vector b, while u has input vector a. Choose a and b to be nonzero, centered, orthogonal vectors of equal norm. With unit LayerNorm scale and zero offset, their normalized vectors remain linearly independent. In one head, set all query and key projections to zero, giving uniform attention. Choose a linear value projection that maps LN(b) to zero and LN(a) to a nonzero vector w; such a map exists by linear independence. Set the other heads' values to zero, choose the output projection not to annihilate w, and set the first-layer MLP to zero.

The native attention output at t contains w/|N(t)|. Every physically available source in the reordered row has value zero, so every nonempty deletion-only choice has zero attention output. The destination residual is b in both cases, since t itself cannot be the missing occurrence. The resulting first-layer states differ. This construction also shows why adaptive deletion alone cannot restore a source contribution when all available value vectors are zero.

Thus containment is sufficient for parameter-independent restoration and its failure permits counterexamples to that guarantee. It does not establish necessity for a particular trained model: zero projections, ignored predecessors, cancellations, or other representational redundancy can make missing context behaviorally irrelevant.

## Proposition 2: the complete restorable layout class has 105 members

Restrict attention to physical permutations of the eight dictionary occurrences satisfying Ki before Vi for every i, with Q appended last.

**Count of all layouts.** Of the 8! permutations, each of the 2^4 independent within-pair orientations is equally represented. Hence there are

\[
\frac{8!}{2^4}=2520
\]

own-key-before-value layouts.

**Characterization.** Every value's native predecessor set is physically available if and only if

\[
V_0<V_1<V_2<V_3
\]

in physical order.

For sufficiency, take Vi. Every earlier Vj, j<i, is already visible, as is Kj because Kj precedes Vj. Ki is visible by the own-pair constraint. Thus all of N(Vi) are present. For necessity, if value order is not preserved, some j<i has Vj physically after Vi. Then Vj∈N(Vi) but Vj∉C_L(Vi). At least one value fails containment. □

**Count of the characterized class.** Simultaneously relabeling the four key/value pairs preserves the own-pair constraint and maps any specified relative order of the values bijectively to any other. The 24 possible value orders therefore partition the 2520 layouts into equal classes. The preserved-order class has

\[
2520/4!=105
\]

members, leaving 2415 outside it. This quotient is a symmetry count over labeled occurrences; it is not an assertion that trained position-dependent models behave identically under pair relabeling.

Within the 105 layouts, values never acquire later-value predecessors. Their only extra native-relative sources are visible keys Kj with j>i. Deleting exactly those key edges therefore invokes Proposition 1 for all four values. Outside this class, deletion cannot universally restore all native value states: at least one native value predecessor is missing. Other native keys may also be absent in these boundary layouts.

## Proposition 3: final-query first-layer invariance

Q is physically last in every layout. Its input residual and its source-occurrence set—all eight dictionary tokens plus itself—are unchanged. Its first-layer attention sums therefore agree by the same source-permutation argument. Since the query row is unedited, its complete first-layer state equals native for arbitrary weights across all 2520 layouts and across the value-row masks considered here.

This statement is about the first layer. The second-layer query attends to updated first-layer states, which can differ across layouts.

## What remains after the primary repair: a key-state decomposition

Fix a primary layout and its correct guard. At the input of layer 2, all four value states and Q's state agree with native. Only the four key-occurrence states may differ. The two-layer architecture performs all attention rows in parallel from these layer-input states. Consequently altered layer-2 states at other destinations cannot feed back into the layer-2 query. The only remaining route from the changed first-layer representations to the final query output is through layer-2 attention to the four key occurrences.

This is a statement about **key-token positions**, not just the attention K projection. Their changed residual states can alter both their layer-2 attention keys and their layer-2 attention values.

For one layer-2 head at Q, let r=n denote native and r=g the guarded layout. The head query is shared. Partition sources into A={K0,K1,K2,K3} and B={V0,V1,V2,V3,Q}. Write a_{r,s} for the normalized attention weights and v_{r,s} for the head's projected value vectors. Define

\[
\alpha_r=\sum_{s\in A}a_{r,s},\qquad
\kappa_r=\frac{\sum_{s\in A}a_{r,s}v_{r,s}}{\alpha_r}.
\]

Here κ is a conditional mean of projected **values at key-token positions**, not an attention-key vector. For finite logits, both groups have strictly positive mass. The conditional mean over B is shared:

\[
\mu=\frac{\sum_{s\in B}a_{r,s}v_{r,s}}{1-\alpha_r}.
\]

Indeed, all non-key source states, their logits, and their projected values are unchanged. A different global softmax denominator rescales all their weights equally and cancels when conditioning on B. Therefore

\[
z_r=(1-\alpha_r)\mu+\alpha_r\kappa_r,
\]

and the exact per-head difference is

\[
\boxed{z_g-z_n
=\alpha_g(\kappa_g-\mu)-\alpha_n(\kappa_n-\mu).}
\]

The decomposition includes both changed attention mass and changed content at key positions. It does not say that one of those terms dominates. The full attention-update difference is the output projection applied to the concatenated head differences; its affine bias cancels. The query's incoming residual is shared, but its subsequent MLP, final normalization, unembedding, and softmax need not preserve small differences in a behaviorally harmless way. A near-tied argmax can flip after an arbitrarily small logit change, and unconstrained learned gains prevent a universal small-effect bound from this identity alone. An empirical probability or accuracy guarantee requires additional bounds or measurements.

The shared-μ derivation specifically assumes restored value and query states. It must not be applied unchanged to arbitrary sham masks or boundary layouts where value states can differ from native. A full-state oracle that also restores the keys does make every layer-2 query input agree; that output identity is an algebraic control, not evidence for the empirical sufficiency of the partial guard.

## Proposition 4: own-key-and-self invariance is not native restoration

In every own-key-before-value layout, Ki and Vi are both available at Vi. Restricting each first-layer value row to exactly {Ki,Vi} makes its complete first-layer state identical across all 2520 layouts, for arbitrary weights. The proof is again Proposition 1's source-set argument, now using the same two-element reference set in every layout rather than the native predecessor set. The appropriate reference is native physical order with the **same own-key-and-self mask**.

Except at V0, this two-element source set generally omits native predecessors. The resulting state need not equal the ordinary native state; the counterexample construction above applies to an omitted source. Layout invariance therefore implies neither native-state identity nor correct answers. Key states remain layout-dependent and can affect the final query through layer 2. Even if they did not, a layout-invariant value representation could still be unsuitable for the task.

## Empirical interpretation

The mathematical identities justify controlled interventions and define their failure boundaries. They do not predict success for the trained models by themselves. The empirical E5 claims concern accuracy despite the uncorrected key states, closeness to native output probability, specificity relative to the complete matched-mask family, and separately registered boundary and edge effects. None follows solely from the source-set equalities. In particular, failure of a sufficient structural guarantee is not proof that a trained model needs the missing source, and successful repair is not proof of a unique learned algorithm or necessity of every retained or removed edge.
