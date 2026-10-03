# Experiment 2 prospective implementation and input review

**No critical blocker found for the planned discovery-only execution.** This review is not authorization to run confirmation before remote preregistration. I reviewed the model hook, runner, condition grid, synthetic tests and data preparation, reconstructed the forbidden set, and inspected the three input splits without model evaluation. Trained checkpoints and Experiment 2 intervention outcomes were not inspected or executed. All model tests used fresh random weights and unrelated synthetic inputs.

## Input audit

Reconstructed **2,838,765 distinct forbidden input sequences** and verified its recorded SHA-256 identity: `b009d184a0d838d877b5d7aa0aa7559bdd78eacb358457aee240b2d81a5a0844`.

| Split | Recipients | Distinct inputs including both donors | Forbidden overlap | Other-split overlap | Internal duplicates |
|---|---:|---:|---:|---:|---:|
| Calibration | 512 | 1,536 | 0 | 0 | 0 |
| Discovery | 1,024 | 3,072 | 0 | 0 | 0 |
| Confirmation | 2,048 | 6,144 | 0 | 0 | 0 |

Every input has nine valid tokens, four distinct keys and four distinct values. Recipient targets exactly match the value associated with the query. Both donors preserve query identity and queried-pair position. Every active donor changes the correct answer; every matched donor preserves the recipient's original answer. Active/matched donors have identical keys. Independently reconstructed each matched donor's value replacement or swap and confirmed its frozen bytes.

Matched-donor construction changes two value positions relative to the active donor when the original answer already occurs elsewhere. This happens in 94 calibration, 236 discovery and 501 confirmation cases; remaining matched donors change one value position. Thus these are valid same-answer controls, but are not uniformly one-position or norm-matched controls. Their internal effects need not be zero.

Verified every split-file hash and each cumulative forbidden-set hash. The preparation record reports one rejected discovery bundle and no calibration/confirmation rejections. This review checks the accepted data and its semantic/separation invariants; it does not independently regenerate and replay every candidate rejection. Machine-readable results and source hashes are in `outputs/experiment2/preflight_data_verification.json`.

The forbidden-set builder covers the disclosed existing and new training streams, reused training validation, Experiment 1 discovery, its selected confirmation inputs/donors and original candidate stream, plus disclosed trained-model smoke inputs. The separation criterion is exact full-token-sequence identity, not exclusion of overlapping tokens or partial dictionaries.

## Condition and novelty audit

The grid has **53 discovery conditions**, **92 novel conditions**, and their concatenation of **145 confirmation conditions**. The novel families contain 36 pairs, 24 triple-corruption/sole-head-rescue cases and 32 cross-layer cases. I canonicalized rescue as corruption of the complementary heads: there are no duplicate novel interventions and none is algebraically equivalent to a discovery intervention. Discovery contains no sole-head rescue outcomes.

L1 patches select all four value positions `[1,3,5,7]`; L2 patches select final query position `[8]`. These sets are fixed by the grid. The downstream head identifiers and corruption methods match the declared family definitions. Primary family weighting and forecast/scoring gates remain to be verified when those files are finalized.

## Runner and hook audit

All eight supplied synthetic tests passed on rerun. They cover exact unpatched equivalence to the original model/state dict, the new pre-output-projection head-output cache, position-specific calibration means, frozen-mean immutability, no-ops, all/none rescue endpoints, rescue/complement equivalence, donor self-replacement, batching and input validation.

The important cross-layer behavior is implemented correctly. After creating an upstream patch, the runner forwards with that patch to obtain the current downstream cache. It constructs the selected downstream-head replacement from this updated cache and reruns with the same upstream patch retained. Unselected downstream heads therefore remain responsive to the upstream intervention. The test suite explicitly checks that these heads differ from clean behavior when expected and equal the upstream-only cache.

I additionally compared the actual **mean** and **resample** cross-layer operations against independent direct masking at the live L2 attention-projection input, using a separately seeded random model and synthetic inputs. Target probabilities matched exactly in both cases. This complements the supplied zero-corruption live-hook test; no confirmation input or trained model was used.

Calibration accumulates activations in float64 and freezes position/head means in float32, separately by layer/component. Rescue intentionally copies the declared clean heads from the untouched recipient and supplies the declared corruption to the remaining heads. At most one operation per layer/component is allowed, preventing ambiguous same-component overwrites. The runner is a computation primitive and does not enforce data-phase boundaries itself; its caller must enforce discovery versus confirmation.

## Preservation and remaining freeze requirements

All **45 Experiment 1 frozen hashes** still match its v2 lock. This review created only Experiment 2 review artifacts; it changed no Experiment 1 source, forecast or result.

Before confirmation, independently check and freeze the six checkpoints, calibration tensors, discovery outputs, all quantitative forecasts, family-balanced scorer, exact inputs, full condition grid, decision gates and remote lock. Confirmatory inputs have been inspected only for allowed input invariants in this review. No forward passes on those inputs were run. The distinction between contribution under artificial corruption and strict natural necessity remains essential; matched controls, rescue sufficiency and cross-layer effects do not remove that limitation.
