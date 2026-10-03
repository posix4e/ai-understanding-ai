# Skeptical design review

This is a prospective design critique. No confirmatory results were inspected or generated. The two proposed mechanisms are hypotheses supplied before the experiment; finding support for one is hypothesis discrimination, not discovery of an unanticipated mechanism.

## The discriminating contrast

Let a recipient contain `(a,A), (b,B), (c,C), (d,D)` and query `a`. A key-swap donor contains `(b,A), (a,B), (c,C), (d,D)` with the same query and value tokens. Its correct answer is `B`, while the recipient's is `A`. A value-swap donor instead contains `(a,B), (b,A), (c,C), (d,D)` and also answers `B`.

* A mechanism that writes each key into the following value position predicts that key-swap information can move through layer-1 value-position residuals and layer-2 keys at those positions. Patching both swapped positions provides a cleaner test than patching only one: a one-position patch can create duplicate or absent matches.
* A mechanism that puts a positional pointer in the query predicts that a key swap can change the layer-1 query representation and layer-2 query vector; transferring that query vector should transfer the pointer when the remaining dictionary positions are aligned.
* The value-swap donor tests content transfer. Layer-2 values at the exchanged positions should carry the changed content in either mechanism. This is a useful positive control, but by itself does not distinguish the competing routing mechanisms.
* Use the same recipient/donor pair across intervention sites, and report each seed separately. A mixture or different mechanism across seeds is a valid outcome.

## Confounds and interpretation limits

1. A whole residual contains token identity, positional information, and contextual changes. A successful layer-1 residual transplant is evidence for a causal mediator at that site, not proof that its represented variable is a key or a pointer. L2 Q/K/V contrasts and donor manipulations strengthen the interpretation but do not make the representation semantically unique.
2. A layer-2 query can change under a key-writing mechanism too; a value-position key can change under a pointer mechanism too. Treat the hypotheses as approximate restricted models, not logically exhaustive architectural classes. Neither a single attention map nor one transplant establishes exclusivity.
3. Key swaps must preserve value identities, positions, query token, sequence length, and all other tokens. Value swaps must preserve keys. Use distinct values within each dictionary so recipient and donor targets differ. Queries and matching-key locations must be balanced.
4. Specify simultaneous versus sequential patches, exact tensor indices, pre/post-layer-normalization boundary, and whether donor tensors come from one untouched donor run. Subsequent recipient computation must use the patched activations. Mixing donor output from a previously patched forward pass invalidates the intended intervention.
5. A full-vector swap can create an unnatural activation combination. Matched donors and a no-op test establish some validity, but cannot eliminate distribution-shift explanations. Report patch distances or norms descriptively if feasible; do not infer natural necessity solely from transplantation sufficiency.
6. Position or token shortcuts can solve insufficiently randomized data. Randomize dictionary permutations, query position, key/value assignments, and values independently; check all four queried pair positions. Validation dictionaries used to decide training adequacy must be disjoint from discovery and confirmation examples.
7. Three seeds are a small pilot. Thousands of examples do not become thousands of independent training replications. Report uncertainty across examples conditional on each model and a separate descriptive seed range; a pooled example-only interval overstates robustness across trained models.

## Minimum control set

* Recipient-to-itself no-op patches at every evaluated site: patched logits should equal unpatched logits within a declared numerical tolerance.
* Wrong-position patches with a prespecified position-selection rule and matched number of patched positions when possible.
* Wrong-component patches at the same positions, without calling a component a negative control if either theory predicts it can affect the task.
* Matched irrelevant-swap donors that leave the queried association unchanged, preserving swap count and donor construction.
* Unpatched recipient and donor accuracy, target probabilities, and recipient-versus-donor target logit contrasts. Include failures in the primary analysis; any accuracy-filtered secondary analysis must be predefined.
* A strong content-transfer positive control using value-swap donors. Failure of all interventions is inconclusive if training accuracy or patch implementation is inadequate.

## Prediction and scoring recommendations

Freeze an executable predictor calibrated only on discovery runs. State whether predictions concern individual cases or condition-level means. At minimum commit a numeric expected recipient/donor target contrast for every seed and intervention condition, the metric and units, and the scoring rule. Rank ordering or verbal signs alone is not a quantitative forecast.

Use a bounded donor-versus-recipient probability contrast for a stable primary effect measure, or explicitly guard against near-zero denominators in normalized recovery. Report raw logits/probabilities alongside any normalized effect. Useful baselines are zero intervention effect, complete donor-behavior transfer, and a prespecified discovery-calibrated condition mean. Evaluate the frozen mechanistic predictor against those baselines with the same cases and loss.

Specify before confirmation the main hypothesis comparison, positive/negative expectations, exclusions, missing-case handling, and decision threshold. Treat other contrasts as exploratory. Bootstrap paired recipient/donor/intervention examples together within seed; retain the entire intervention family for each sampled example. Avoid selecting only the most favorable head, seed, position, or metric after confirmation.

## Audit checklist before releasing confirmation

- [ ] Training is finished for the prespecified seeds and checkpoints are frozen.
- [ ] Discovery, training-validation, and confirmation generators have distinct fixed namespaces/seeds; the split unit and possible overlap are documented.
- [ ] No confirmatory model forward pass, metric, prediction-error calculation, plot, or preview has run, including during script smoke tests.
- [ ] The full predictor, parameters, numeric forecast table, protocol, condition list, evaluation code, checkpoint hashes, and data/generator hashes are saved and committed.
- [ ] Repository and runtime dependencies sufficient to rerun are recorded; randomness and device are specified.
- [ ] A remote read verifies the exact preregistration commit and hashes before the confirmation command is enabled. A local commit or successful push alone does not demonstrate remote verification.
- [ ] Record remote verification evidence and its time before confirmation start; keep raw result artifacts separate from the preregistration commit.
- [ ] The frozen evaluator applies all declared patches and controls, and discovery-only tests establish causality, no-op equality, donor isolation, shapes, and indexing.
- [ ] After confirmation, report every prespecified condition and seed, all failed controls, uncertainty, baselines, and any deviation. Do not silently revise forecasts or training after looking at outcomes.

## Recommended claim boundary

A successful pilot would show that a frozen explanation made useful out-of-sample predictions about specified interventions in these trained models, and would favor one restricted routing account over another within the intervention family. It would not establish unique mechanistic recovery, arbitrary-model interpretability, or human-independent discovery of the hypotheses.
