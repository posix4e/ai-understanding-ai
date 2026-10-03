# Prospective skeptical design review: Experiment 2

This review precedes Experiment 2 discovery and confirmation. No Experiment 2 model forwards or outcomes were inspected or generated for this review. Experiment 1 artifacts and its failed-v1/successful-v2 distinction must remain unchanged.

## What the experiment can identify

The useful question is whether a discovery-only account predicts the effects of removing or restoring specified attention-head outputs, including combinations not observed during discovery. A head's contribution under zero, mean or resampled replacement is an operational causal quantity. These interventions do not by themselves establish that the head is naturally indispensable, uniquely implements a semantic operation, or remains sufficient in arbitrary contexts.

Measure the output `z` of each attention head after weighted value aggregation and before the shared output projection. Zeroing all such heads retains the output-projection bias, residual bypass and MLP. Name the intervention accordingly; it does not remove an entire transformer layer. Verify that adding this cache/patch hook leaves all state-dict keys and unpatched logits unchanged.

Head labels are arbitrary across independently initialized models. Report each model's head identities and discovery pattern; do not treat head 0 in one model as the same learned circuit as head 0 in another. Existing seeds 0–2 are an extension of known models; new seeds 3–5 provide a separate, more informative replication stratum. Keep every prespecified model, including inadequately trained or mechanistically inconvenient seeds.

## Novelty and exact intervention equivalences

For a fixed corruption tensor, corrupting all heads and restoring one clean head is exactly the same intervention as corrupting the other three heads. It must not appear in discovery if three-head interventions are advertised as unseen. Likewise, a single-head ablation is equivalent to a three-head rescue from the same fully corrupted tensor.

The revised plan resolves this: discovery contains only clean behavior, single-head corruption and all-head corruption; no rescue cases. Confirmation reserves:

| Family | Complete fixed grid | Count |
|---|---|---:|
| Two-head corruption | Six pairs × two layers × zero/mean/resample | 36 |
| Sole-head clean rescue / three-head corruption | Four retained heads × two layers × zero/mean/resample | 24 |
| Cross-layer corruption | Four L1 heads × four L2 heads × mean/resample | 32 |
| **Novel scored total** | | **92** |

Call the second family both names where useful, but count it only once. Freeze the full grid before discovery results; do not select especially promising heads or a new cancellation condition after seeing discovery. Familiar conditions and controls should still be measured during confirmation, but should not dominate the primary novel-condition score. Specify separately whether matched-resample versions enter any secondary grid; they should not be silently added to the 92-condition primary set.

## Cross-layer semantics are a critical implementation constraint

A downstream patch assembled from an untouched clean recipient cache would reset every unselected L2 head to clean behavior and erase upstream L1 effects. That would test a different intervention from changing only the selected heads.

The planned implementation instead constructs the upstream patch, runs that patched model to obtain the current downstream cache, modifies only selected downstream heads in that cache, and reruns with the identical upstream patch plus the downstream patch. In this deterministic, dropout-free model this is equivalent to modifying only selected heads at the live downstream hook. Require a random-weight check against a direct live-hook oracle. Unselected downstream heads must equal the upstream-only run's values, not necessarily the clean run's values.

For rescue, the declared rescued heads may intentionally use untouched clean activations. The remaining corrupted heads must use the same corruption tensors as the corresponding all-head-corrupted comparator. Do not change donor draws between the all-corrupted and rescue conditions. Within a layer, combining zero/mean/resample corruption and rescue must retain their exact union rather than accidentally overwriting an earlier patch.

## Corruption and donor controls

Use position- and head-specific means from a separate 512-case calibration set, independently for each model. Freeze the calibration inputs and resulting tensors. Means must not be re-estimated from discovery or confirmation, and recipient targets must not select the mean.

Define the L1 position set exactly: all four value positions versus only the queried value position are materially different interventions. Use the identical declared set across the head grid. L2 interventions affect the final query position.

The proposed resampling comparison holds query identity and queried-pair position fixed. Active donors give a different answer; matched donors preserve the original answer while altering other context. Freeze a construction that retains four distinct keys and values. If constructing the matched donor by replacing the active donor's queried value, ensure that value does not already occur elsewhere, or specify an exact swap/rejection rule. Otherwise repeated values would change the task distribution and donor interpretation.

Matched-output donors are informative about sensitivity to context versus answer content; they are not automatically negative controls. A same-answer context can still corrupt an internal route. Do not force their forecast to zero by definition. Zero replacement is off-distribution; mean replacement can also be atypical; resampling preserves marginal donor values but disrupts their joint compatibility with recipient activations. Agreement across corruption schemes is stronger evidence than any one alone. Report replacement distances/norms descriptively if available; do not call the donors norm-matched without checking.

Required implementation controls are clean-to-clean replacement, all-head clean rescue, no-op at each layer and combined layer pair, unchanged donor caches, masked-head selection, head-order/union invariance, unpatched equivalence to Experiment 1, and causal recomputation for cross-layer patches. Clean all-head rescue should reproduce the clean output; sole-head rescue has no such guarantee. These controls validate execution, not the mechanistic hypothesis.

## Forecasting and baselines

Separate two claims: practical accuracy of the frozen explanation, and incremental predictive value over simpler discovery-calibrated rules. Report both even when only one succeeds. The predictor should formulate its account from discovery evidence and be permitted to predict uncertainty or fail; the coordinator should not dictate its scientific conclusion.

Single-head and all-head results do not uniquely determine pair or triple behavior. If single removals do little and all-head corruption destroys performance, multiple redundancy/threshold mechanisms remain compatible with discovery. A precise subset forecast is then an extrapolation, not a deduction. This is precisely what the reserved grid can test.

Freeze several useful comparisons, using the same discovery information:

* **Additive change:** sum constituent single-head changes from the clean probability, with a declared probability clipping rule.
* **Multiplicative retention:** multiply single-head probability-retention ratios relative to clean behavior; freeze denominator protection and clipping. If chance-adjusting, prespecify whether chance means 1/16, 1/4 or an empirical corrupted baseline, because these differ substantially.
* **Count interpolation:** ignore head identity and interpolate between clean, average single-head corruption and all-head corruption. This is particularly important when all heads are redundant and head-specific rules add little.
* **An optional stronger head-specific saturation/redundancy model:** fit only from discovery and calibration, with the functional form, regularization and fitting objective frozen before confirmation. Do not introduce it only after it loses or wins.

Do not describe these as interchangeable baselines. Additive probability changes can fail mechanically at floors/ceilings; a mechanistic account's advantage over that baseline alone is weak evidence. Report every prespecified comparator, including whichever performs best, and require a declared comparison criterion to claim added value. Because rescue/triple effects are not available in discovery, a baseline requiring sole-head rescue measurements would violate the split.

For cross-layer baselines, use only available discovery single-head and all-head statistics. Do not calibrate using the reserved within-layer pair/triple results. Output probability is a clear primary endpoint; original-answer accuracy and a declared finite logit contrast can be secondary. Apparent nonadditivity on probability or accuracy scales can arise from saturation and does not alone identify a mechanistic interaction.

## Scoring and uncertainty

Use three equally weighted novel families—pair, triple/sole-rescue, and cross-layer—rather than allowing differing condition counts to set the scientific weighting implicitly. Within each family, average over the full fixed layer/head/corruption grid. Exclude clean/no-op and structurally trivial controls from the primary forecast score while reporting them separately. Keep the familiar-intervention score visible as a secondary calibration check.

Freeze the primary forecast endpoint, units, mean-versus-case prediction level, error tolerance, success gates, baseline loss and family weights. Do not score each mean forecast as a claim about every individual example. Show all 92 forecasts per model, all errors, family-specific losses and failure identities. Report both existing-model and new-model results; do not replace a failed new model with another seed.

Resample whole example groups with recipient/donor pairing and all interventions together. For any aggregate over models sharing those inputs, use the same sampled case indices across models; independent per-model resampling would lose that dependence. Per-model case bootstrap intervals remain conditional on fitted means, discovery and checkpoints. Six models still provide limited evidence about training variability; report seed outcomes and ranges explicitly. Use Wilson intervals for accuracy near 0%/100%. Practical bounds are not calibrated predictive intervals, and joint gate success is not 552 independent tests.

## Input separation and prospective audit

Before any confirmatory forward, freeze exact recipient/donor bytes, calibration means, checkpoints, complete numeric forecasts and all evaluation/scoring code. Prepare the disjoint inputs before registration so a predictable finite-sample overlap does not force another post-registration repair.

The forbidden-set specification should include all relevant old and new training streams, reused validation, Experiment 1 discovery and frozen confirmation recipient/donor inputs, and relevant trained-model smoke inputs. Keep Experiment 2 calibration, discovery and confirmation disjoint as full groups, including every active/matched donor. Specify accepted-recipient duplicate handling and exact batch-dependent generator behavior. Record rejection counts/reasons and forbidden-set identity. Do not generate confirmatory model outputs while preparing or auditing inputs.

Remote verification must include actual selected input bytes and fitted calibration tensors, not merely seeds or metadata. Preserve Experiment 1 files; use separate Experiment 2 manifests, locks and result directories. The blinded predictor may inspect only authorized discovery/calibration evidence and implementation, then must freeze all forecasts before confirmation is released. Shared-filesystem procedural blinding remains a limitation and should be described honestly.

## Claim boundary

A favorable result would demonstrate useful prospective prediction of corruption and rescue behavior across a prespecified head-combination grid, including additional trained models. It would strengthen evidence beyond Experiment 1's single discriminating cross-donor comparison. It would still be conditional on these interventions and tasks, and would not show unique circuit semantics, strict natural necessity, or general AI self-understanding.
