# Organ route strength: design only

**DESIGN_READY_RUNNER_NOT_IMPLEMENTED.** This folder contains a fixed metadata
schedule and scalar scoring arithmetic. It contains no model runner, registration,
raw-tensor auditor, resource preflight or scientific result. Passing the invented
tests is not a successful experiment or complete preregistration.

The question is whether attenuation of the same supplied-answer displacement
changes the contribution of the implemented routing policy. This is a controlled
test of the existing engineered channel, not a test of semantic composition,
energy superiority, natural mediation or expert IDs alone.

## Fixed panel

Reuse all 96 exposed cells (24 language contexts × four worlds), the pinned
Granite revision, and the original target labels. Read only the census plan and
its hash-bound tail plan to create `plan.json`. The strengths are exactly 0,
1/16 and 1. In every cell the order is:

| Strength | Arms |
|---|---|
| 0 | N, S |
| 1/16 | N, S, F0 |
| 1 | N, S, F0 |

N executes native routing. S replays that condition's N routing tuple. F0
replays the original census B tuple. All experts recompute from current hidden
inputs. All four tail layers and the full sequence are controlled together;
gates, support, grouping and kernel numerical effects belong to the intervention.
There are 768 suffixes and 3,072 decoder-block executions. No prefix, reader,
energy, writer, fit, tokenizer, generation or provider calls are proposed.

## Interpolation and admission contract

Start from the original pre-writer hidden tensor. Multiply its saved FP32
displacement by the fixed strength in FP32, cast that displacement to BF16, and
perform the original native BF16 addition at the final row. Copy all other rows.
Do not interpolate the already rounded B/W returned hidden tensors.

Strength 0 must reproduce B and strength 1 must reproduce W exactly. All 192
native endpoint traces and 288 same-condition full-logit/internal-trace shams
must match before scientific scoring. Retain every cell. Do not require
cross-policy nonfinal equality; measure its drift. Validate delivered policy
against its donor, not the live logits that the clamp intentionally overrides.
Failure is not permission to select a strength, refit, or remove cells.

These tensor/provenance checks are requirements for future code, **not implemented
by `metrics.py`**. Raw census seals, parameter/ownership controls, independent
audit, source review and execution registration are still required.

## Arithmetic

For the fixed target, J is its logit minus the mean of the seven other logits.
The route effect is J(N) − J(F0), so positive helps. The amplitude contrast is
route-effect(1) − route-effect(1/16). Retain all seven target-other contrasts,
best-other margin, strict unique choice, correctness, ties and every per-cell
effect. Ties are unresolved and incorrect; nonfinite data is structural failure.
Report per-world and pooled means and sign counts without p-values or treating
the 96 cells as independent language contexts. One attenuation does not map a
continuous dose-response curve or establish saturation.

`metrics.evaluate_scores(plan, records)` accepts exactly 768 `{id, scores:[8]}`
records and returns arithmetic with `implementation_admitted: false`. It cannot
certify a scientific archive or bypass the future raw-control admission.

## Bounds and preparation

Proposed limits: 900 seconds total, 10 GiB process-tree RSS, 10 GiB MPS driver
allocation, 2 GiB archive and at least 8 GiB free storage. A later resource
preflight must establish feasibility; these limits do not guarantee it.

Run only the invented arithmetic/schedule tests:

```sh
python3 -B -m unittest discover -s research/moe-mechanisms/organ_route_strength_v1 -p 'test_*.py' -v
```

Create the design file exclusively from metadata:

```sh
python3 -B research/moe-mechanisms/organ_route_strength_v1/protocol.py --write-design research/moe-mechanisms/organ_route_strength_v1/plan.json
```

Neither command loads a model or imports an old frozen runner.
