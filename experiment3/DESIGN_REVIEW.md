# Experiment 3 prospective skeptical design review

This review precedes all trained-model Experiment 3 forwards. No new interventions were executed for this review. The experiment tests a distinct causal hypothesis about supplied positional representations; it is not an out-of-model transfer test of the Experiment 2 surrogate.

## Decisive hypothesis and controls

The useful claim is **answer-specific rebinding by positional codes**, not merely sensitivity to position or failure on a changed layout. Canonical-coordinate rescue alone is insufficient: it could simply restore the trained key/even-position and value/odd-position roles.

The proposed complete permutation grid resolves this confound. In grouped physical order `K0 K1 K2 K3 V0 V1 V2 V3 Q`, keep query position ID 8 and use every trained ID 0–8 exactly once. For a permutation `pi` of four pair slots:

* **Coherent:** key i receives ID `2*pi[i]`; its own value receives ID `2*pi[i]+1`. Predict the original intended answer.
* **Value-only:** key i receives ID `2*i`; value i receives ID `2*pi[i]+1`. Predict the value indexed by `inverse(pi)[query_pair]`, which may deliberately conflict with the external dictionary's intended answer.

Both controls preserve role parity and the position-ID multiset. All keys physically precede all values, so none of the hypothesized redirected key-to-value links is causally masked. A switch toward the exact named alternative distinguishes rebinding from nonspecific degradation. Coherent permutations test whether the association follows the assigned slots rather than a privileged original slot label.

The two identity mappings are the same canonical-grouped condition and must be counted once. The resulting grid is 23 nonidentity coherent + 23 nonidentity value-only + canonical grouped + native grouped + original native + original no-op = **50 conditions**. The nine derangements supply the primary comparisons. The other 14 nonidentity permutations are secondary; distinguish their moved-query cases from fixed-point cases, because fixed points leave the predicted and intended answers identical.

## Agreed numerical gates and uncertainty

Use all six existing models, the same 2,048 dictionaries per condition/model, and exactly 512 accepted queries at each pair index. There is no new training, discovery intervention phase, parameter fitting or head selection.

The frozen primary claim requires:

1. **Every seed × each of the nine derangements:** named slot-predicted answer accuracy >=0.90 under value-only IDs.
2. **Every corresponding coherent condition:** original intended-answer accuracy >=0.95.
3. **Every seed:** the lower endpoint of the paired, query-stratified 95% bootstrap interval for the mean softmax contrast `P(slot-predicted value) - P(original value)`, averaged equally over the nine derangements, is strictly above 0.80.
4. **Validity:** original native accuracy >=0.95 and the stated original no-op maximum target-probability discrepancy <=1e-6, with finite complete outputs.

“Named slot-predicted answer accuracy” measures agreement with a behavioral prediction, not correctness on the external dictionary task. In a derangement these two answers differ. The 0.80 gate is explicitly a **softmax-probability** contrast; an accuracy difference would largely duplicate the 0.90 directional-accuracy gate.

Report all 54 primary seed/permutation cells rather than only their average. Use individual Wilson intervals for Bernoulli accuracies, including observed 100%. These intervals are not simultaneous joint confidence bands. The point-gate conjunction across cells is the declared replication rule; do not describe it as a separately multiplicity-adjusted confidence statement.

For the probability contrast, resample complete dictionary rows **within each of the four query strata**, retaining 512 rows per stratum. Share the sampled indices across every map and model. Within each resampled row retain all nine derangements before averaging; 18,432 row-map observations are not independent cases. Freeze bootstrap count/seed before outcomes. All six seeds must pass; failure cannot be replaced by a favorable pooled model average.

Native grouped failure is neither required nor sufficient for rebinding. Record it descriptively. Any additional numeric forecast tolerance, such as 0.15, is a distinct forecast-adequacy report and cannot substitute for the causal gates. Execute the entire frozen grid before interpretation; do not stop when a desired contrast first passes.

## Baselines and interpretation

A semantics-only account predicts the original dictionary answer under both coherent and value-only maps. A parity-only symmetric account cannot distinguish values sharing the odd-position role and assigns equal mass to the four displayed values. The slot-binding account names a particular value from the inverse permutation. These are explicit competing behavioral rules, not claims that the experiment defeats the strongest statistical baselines in interpretability research. If using proper scores, freeze the forecast distributions and smoothing so a nominally deterministic prediction does not create arbitrary infinite log loss.

Use output behavior as primary evidence. Attention to the implied key/value can provide corroboration, but an attention-map change is not a causal mediator test. Do not choose a favorable head after viewing Experiment 3. Either report all heads or use an independently fixed selection rule. A strong behavioral result plus matching attention still does not establish a unique representation.

Even a complete success cannot distinguish an arbitrary learned absolute-slot lookup from computing “previous coordinate” in the supplied learned position-code system. It establishes dependence on assigned positional coordinates over physical adjacency under the tested edits. The input representations are externally altered: recovery is **assisted layout transfer**, not spontaneous layout generalization.

## Prior work and novelty boundary

Position sensitivity itself is not a new discovery. [Singh, Mishra and Raut (2026)](https://arxiv.org/html/2607.18759v1) study fixed-offset retrieval and absolute-position pinning, including failure at out-of-range untrained coordinates. [Ruoss et al. (2023)](https://arxiv.org/abs/2305.16843) study randomized positional encodings for length generalization. These references constrain the novelty claim; they are not an exhaustive literature review establishing priority for this proposal.

Here every edited coordinate is already trained and the multiset of coordinate embeddings is held fixed. The prospective contribution, if supported, is the specified **answer-by-answer permutation law under conflicting in-range position labels in learned content-addressed lookup**. That is a narrower result than “absolute positions matter.” A grouped-layout success alone does not establish its boundary across arbitrary causal layouts. Testing additional layouts should be a separate full preregistration, with label-implied key visibility defined in advance; in partially interleaved layouts some redirected links may be blocked by the causal mask.

## Data and no-fishing safeguards

The data generator's accepted-row query assignment correctly gives exact balance without looking at outcomes. Keep four distinct keys/values, consistent targets and a unique dictionary per accepted row. Audit original and grouped inputs, selected-file hashes, prior-data exclusion, and declared dictionary-level versus full-query-sequence novelty before freeze. The final generator excludes all 96 interleaved representations of each association dictionary (24 pair orders × four queries) against the prior input set, and excludes duplicate unordered association dictionaries within E3. Independent enumeration confirmed zero collisions for all 196,608 selected variants, with 2,849,517 prior inputs.

Freeze the 24 permutations, inverse-map targets, all 50 specifications, six checkpoints, exact input bytes, prediction rule, thresholds, scorer and phase gate. No alternate maps, head subsets, cohorts, donor rules or thresholds may be selected after a trained forward. Preserve all earlier experiment artifacts and every failure. Perform index/permutation/no-op tests on unrelated random weights first; after remote verification run every condition on the fresh fixed inputs. A result that merely lowers accuracy or rescues canonical coordinates without named-answer redirection does not establish the stronger rebinding claim.
