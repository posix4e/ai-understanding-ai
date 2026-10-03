# Prospective implementation preflight

**Result: the inspected transformer and evaluator helpers pass the architecture, donor-construction, and activation-patching checks below.** These checks establish implementation consistency, not support for either scientific mechanism. They used freshly initialized models and a synthetic pilot-only batch. No trained checkpoint or confirmatory examples/results were loaded.

## Scope and reproducibility

Reviewed `model.py`, `train.py`, `evaluate.py`, `score.py`, and `protocol.json`. Executed the reusable `tests/preflight.py` on CPU with PyTorch 2.14.1, deterministic algorithms enabled, model seed 734821, synthetic data seed 834822 and donor seed 834823. Tested both one-head and four-head configurations, each with two layers, width 64 and four dictionary pairs. Run from the repository root with `.venv/bin/python tests/preflight.py`; it prints machine-readable results and current source hashes to stdout without writing result files. It never calls the evaluator's `run` or `main` entry points. `outputs/preflight_runtime.json` is the earlier architecture-only review artifact and contains historical source hashes.

The expanded joint/cross-donor review tested `evaluate.py` SHA-256 `204d53e77ac92fa25467aa2b70883c33e446f8ef935b454345e06a52ca229e3a` and inspected `score.py` SHA-256 `2d6b404f8e8ad5bb712da87ac26fcfe96aeff3099be7e68f0cfffdde5d6d8f76`. Later edits require rerunning the relevant checks.

## Runtime checks

| Check | Result |
|---|---|
| Task token/class indexing and output dimensions | Passed |
| Modify every possible sequence suffix and compare preceding logits | Exact equality for both configurations |
| Future-token attention weights | Exactly zero |
| Attention normalization | Within absolute tolerance 2e-7 |
| Recipient-to-itself patches at all 18 layer/component sites, using integer, list, slice and tensor indices | Exact output equality in all 144 cases across both configurations |
| Patched cache equals the donor only at selected positions | Passed for every site |
| Donor cache, recipient cache, and input tokens unchanged after patches | Passed |
| Unpatched donor and recipient reruns after interventions | Exact equality |
| Q/K/V equal independently reconstructed projections after attention layer normalization | Exact equality |
| Scaled masked attention, attention output projection, and residual additions reconstructed independently | Exact equality |
| Layer-1 post-residual patch equivalent to layer-2 pre-residual patch | Exact output equality, with a nonzero intervention effect |
| Layer-2 Q patch preserves layer-2 K, V and incoming residual | Exact equality, with a nonzero intervention effect |
| Unknown intervention sites and mismatched donor shapes | Rejected as expected |
| Independent per-row patch selection over all 12 ordered distinct pair selections, rank-3 residual and rank-4 Q/K/V tensors | Exact equality with independently constructed expected tensors |
| All four donor families: exact changed tokens, query preservation and correct queried answer | Passed |
| All 36 evaluator conditions: batched composite patches versus individually selected patches for 11 examples, both model configurations | Maximum absolute logit difference 2.124e-7, below 2e-6 tolerance |
| Composite recipient-to-itself patches for all 36 conditions, both configurations | Exact output equality |
| Last-layer nonquery Q patches and final-token logits | Exact equality, as structurally required |
| Alternative-minus-original probability contrast and accuracy indicators | Passed |
| All 16 joint conditions: independent union oracle, same-component value-plus-query preservation, order invariance and no-ops | Exact equality |
| Joint conditions versus individual per-example patches, both configurations | Maximum absolute logit difference 2.124e-7, below 2e-6 tolerance |
| Cross-donor L2 K from key-swap donor plus L2 V from value-swap donor: selected positions, no-op and per-example equivalence | Passed; maximum logit difference 2.087e-7 |
| Input with both key and value swaps preserves every original key-to-value association and query token | Passed |
| Wilson intervals at zero and perfect accuracy, and reflection symmetry | Passed; boundary intervals are nondegenerate |

## Code findings relevant to the evaluator

1. Layers are **zero-indexed in code**: scientific L1 is layer `0`, scientific L2 is layer `1`. Q/K/V caches have shape `[batch, position, head, head_width]`; they are projections of the layer-normalized incoming residual. `resid_post` is after both attention and the MLP. `attn_out` is after the attention output projection. The default residual patch copies every feature and every head.
2. The model's `positions` selection is shared across batch rows. The evaluator correctly constructs a composite patch tensor from the recipient cache with only each row's intended donor entries changed, then patches the common position set. This construction passed independent exhaustive position-selection and batched-versus-individual model checks.
3. No dropout or batch-coupled normalization is present. Donor caches are detached clones; intervention assignment clones the current tensor. These choices prevent the tested patches from mutating saved donor state.
4. Direct `attn_pattern` patches are applied after masking and normalization. A malformed external donor could therefore introduce future attention or unnormalized weights. Untouched same-shape causal donor runs remain valid. The proposed Q/K/V and residual interventions do not have this issue.
5. The random task generator independently samples distinct keys and distinct values, then a uniform queried pair. Targets use value-class indices; input values use an offset token vocabulary. Four in-context values imply a 25% baseline for choosing a random presented value, while uniform guessing over the full output vocabulary is 6.25%. Approximately 24% validation accuracy therefore does not demonstrate retrieval and cannot support strong mechanistic conclusions.
6. Training validation is a fixed reused pilot set. It is appropriate for training decisions but must not become confirmatory evidence. The generator uses independent pseudorandom streams rather than explicitly excluding duplicate sequences between datasets; document that distinction or add exact overlap checks if claiming strict disjointness.
7. Active donors exchange the queried pair and one other pair; matched donors exchange the remaining two pairs. For matched donors, active patch positions follow those irrelevant exchanges and wrong-position patches use their complement, including the query-associated pair. This is implemented consistently, but matched and active effects are not a perfect same-position comparison.
8. `l2_q_values` is a structural negative control: a Q change at a nonquery position in the final layer cannot change the final-token output. A zero effect verifies locality but does not favor either scientific mechanism.
9. The primary recorded effect is the intervention-induced change in `P(alternative)-P(original)`, with support [-2, 2]. Matched donors retain the original correct answer; the alternative remains the active donor's alternative for comparability. This contrast can miss redistribution among unrelated output classes, so original-answer accuracy should remain visible. The evaluator reports normal-approximation 95% intervals for mean effects and Wilson intervals for proportions, conditional on each fixed model; these are not cross-seed intervals or simultaneous confidence bounds across conditions.
10. The evaluator's joint patch assembly retains previous changes when two sites share a component, so `joint_l1_values_query` correctly transplants both value positions and the final query position. Different-component joint patches and cross-donor K/V patches use untouched donor caches. The cross-donor full-input comparator applies both swaps and preserves associations; the cross intervention itself still combines separate key-only and value-only donor activations.

## Scorer and protocol review

The scorer uses equally weighted squared errors of condition means as its primary forecast loss, with per-example loss reported separately. It retains all conditions from a sampled row together in bootstrap resampling and pairs donor responses with that row. Thus shared recipient/donor variability is preserved. Familiar and novel-combination groups have separate point losses and bootstrap intervals for the AI-minus-empirical-baseline loss gap. The discovery values and frozen forecasts remain fixed during these bootstrap calculations; the uncertainty is conditional on that fitted forecast and each trained model.

The empirical baseline uses discovery condition means for familiar sites and clipped sums of constituent effects for unseen joint and cross-donor conditions. The full-donor comparator applies a fixed transfer rule to donor responses measured during confirmation; it is not itself a discovery-only numeric forecast. Label this distinction when comparing forecast methods.

The protocol prespecifies three models, data seeds, counts, batch size, effect metric, primary K-versus-Q routing contrast, no-op tolerance and bootstrap randomness. Training-audit budgets differ by seed; the report should explain discovery-guided training changes rather than presenting the models as identical fixed-budget replications. Any binary success criterion and subgroup emphasis must be committed before results. The unseen joint and cross-donor condition choices were made after discovery and before confirmation, which supports prospective evaluation of those selected compositions rather than an untouched initial design.

## Remaining checks owned by the evaluator

This review covers donor construction and per-row patch assembly, and statically reviews scoring; it does not certify forecast fitting, empirical uncertainty coverage, exact dataset disjointness, or the remote preregistration gate. Joint/cross-donor assembly checks independently reconstruct the intended patch behavior; they do not invoke the evaluator's checkpoint-running entry point. The evaluator's remote check reads a commit object and checks local file hashes; the lock-creation workflow must separately establish that the frozen files are the exact contents of that remote commit. Freeze final checkpoint and source hashes after any training or architecture adjustment. Keep weak-training failures visible, and distinguish discovery-guided training changes from confirmatory findings. The pilot-only checks do not establish power or mechanistic identifiability on successfully trained models.

The scientific interpretation limitations and prospective audit checklist are recorded in `outputs/review_design.md`.
