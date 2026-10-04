# E7 design review: distinguish representation damage from later graph changes

Prepared 4 October 2026 from the released E6 evidence and frozen source, without pretrained E7 forwards. Recommendations below are prospective design choices, not observed findings. Existing E1–E6 records remain unchanged.

## The scientific question

E6 showed that restoring first-block value and query-suffix states did not restore GPT-2 answers. In its valid float64 repeat, restricted accuracy was 93.16% native, 10.35% grouped, 9.57% with the first-block guard, and 31.25% with guards in all blocks. These results concern one calibrated synthetic lookup format and one checkpoint. They do not identify why the primary repair failed.

E7 can ask whether the remaining failure is associated with altered first-block **key-chunk residual states**, later changes in which token occurrences can communicate, or an interaction between those interventions. A key chunk includes its label and punctuation; restoring it is not an intervention solely on the attention K projection or solely on label semantics.

Use the same GPT-2 checkpoint, exact float32-to-float64 parameter promotion, selected format, tokenizer and scoring definitions as E6. There should be no new model, prompt, head or layer selection during discovery.

## What the proposed factorial identifies

Every factorial cell starts from canonical grouping plus the correct block-0 value guard. Let S restore native residual states at all key-chunk token occurrences immediately after the complete first block. Let G replace the attention graph in blocks 1–11 with the native causal graph, expressed in the grouped occurrence coordinates.

| Cell | State replacement S | Later graph transport G | Interpretation |
| --- | --- | --- | --- |
| 00 | No | No | E6 first-block guard baseline |
| 10 | Yes | No | Empirical sufficiency of correcting the remaining first-block states under the changed later graph |
| 01 | No | Yes | Empirical sufficiency of correcting the later graph despite altered early key states |
| 11 | Yes | Yes | Algebraic full-output restoration control |

With both interventions, all first-block states match the native sequence after aligning token occurrences: the guard restores value chunks, the suffix and prefix already agree, and S restores keys. Every remaining block sees the same states, position information and predecessor sets as native, up to permutation. Induction therefore gives the same final logits, within floating-point tolerance. **Cell 11 recovering native behavior is not a discovery.** In particular, it is not evidence of a newly discovered mechanism or a practical repair.

Graph transport deliberately opens edges to physically later tokens. It restores the original information graph of a fully supplied prompt; it is an oracle intervention, not ordinary causal decoding in the grouped serialization. Native-state replacement also uses a separate native forward as privileged information. Both facts must appear beside any reported recovery.

The four cells identify intervention effects in this constructed counterfactual setting. They do not by themselves identify unique semantic computations, natural necessity, or the fraction of a naturally occurring error attributable to one route. There is no baseline-independent decomposition of responsibility when effects interact.

## Strongest bounded extension: split the graph factor

For this grouped layout, the difference between physical and transported native causality has two exact, disjoint parts:

- **A: add earlier-value to later-key edges.** At key chunk Ki, add every token in Vj for j<i: six chunk-to-chunk edge groups. These are physically future sources.
- **D: delete future-key to earlier-value edges.** At value chunk Vi, remove every token in Kj for j>i: six chunk-to-chunk edge groups.

Within-chunk causal order, prefix and suffix edges remain unchanged. Thus G=A+D. D alone in blocks 1–11, with the fixed block-0 guard, is the E6 all-block guard policy.

With E6's two-token key and value chunks, each category changes exactly **24 directed token-to-token edges per later block**: six chunk pairs times two destination tokens times two source tokens. Reuse the same graph at every declared later block. Writing P for physical causality and T for transported native causality, the four graph settings are physical=P, deletion-only=P∩T, addition-only=P∪T, and transport=T. The eight cells cross these four settings with S off/on. Addition-only intentionally retains the extra physical edges; it is not the transported graph.

I recommend the complete **2×2×2 S/A/D grid**, rather than a growing series of selected controls. It has eight cells, includes the proposed four, and adds only four. It can distinguish the empirical role of restoring missing past-value context at later key destinations from suppressing extra key context at value destinations. This is more informative than saying broadly that later layers matter. It also supplies the previously observed deletion-only policy as a prespecified comparison on new inputs. The all-on cell remains an identity control and is excluded from empirical candidate selection.

Include native and unguarded canonical grouped references. A separate all-block transported-graph condition is another algebraic oracle: applying T from block 0 onward with canonical positions should reproduce native output without a residual patch. It checks a different implementation path and is also excluded from empirical candidate selection. A native-state patch overwritten with the recipient's own current key states must be a no-op. An explicitly supplied physical graph must agree with the ordinary grouped forward; these may be synthetic/runtime validity checks rather than additional scientific cells. A cyclic wrong-key-state donor is optional and must be labeled a mismatched-content control, not a norm-matched control. Prioritize the complete graph decomposition over adding weakly interpretable donor controls.

## Important implementation constraints

The frozen E6 adapter cannot express A or G: it intersects supplied masks with physical causality, and GPT-2 eager attention applies a second physical triangular mask internally. Create a separate E7 oracle path; do not edit the frozen adapter. Override both sources of physical masking only in the declared later blocks, then impose the desired allowed-edge matrix. Do not mistake adding an external mask for reopening an edge already assigned negative infinity internally.

For grouped permutation p mapping physical slots to original occurrence indices, the transported native mask is `allowed[r,s] = (p[s] <= p[r])`. Use full occurrence indices, not just pair indices. Assert exact A/D differences against this matrix. Retain self edges and nonempty rows. Never transport block 0 in the factorial.

Patch the key residual once, after attention, residual addition, MLP and its residual addition in block 0. Later blocks must recompute their activations endogenously for each cell. Do not insert native downstream caches. Independently test the all-on final-logit identity and the two no-ops with random weights before pretrained forwards. Preserve full-vocabulary output or sufficient summaries for the full-output identity, not only the 16 candidate scores. Record float64 and unchanged parameter checks as in E6.

## Discovery and prospective confirmation

Generate 128 new discovery dictionaries, balanced 32 per query position. Exclude all E6 calibration and confirmation association maps under every pair order and query. Freeze the finite discovery grid before running it and release every condition. New E7 confirmation inputs must also exclude discovery association maps and be frozen before any confirmation outcome is generated.

Discovery is allowed to select a hypothesis; it is not a confirmation. Use all discovery rows and report all effects, avoiding selection of only native-correct or repaired cases. Inspecting individual failures is permitted for hypothesis generation, but their later role must be declared. No layer search, new prompt or revised mask enters the same discovery grid after outcomes are seen.

Before fresh confirmation, choose at most one main empirical candidate among the non-guaranteed cells or one specifically predicted factorial contrast. Write its expected direction, effect size, supporting comparator, numerical forecasts and failure criteria. Keep the full finite grid in confirmation so the selected contrast remains assessable. If discovery produces no substantial, interpretable effect, report that result; the guaranteed control cannot be promoted to the candidate. A subsequent expanded investigation requires a new disclosed design.

## Scoring and claim gates

Retain restricted accuracy, unrestricted next-token accuracy, raw target probability, conditional target probability and candidate mass for every cell and query stratum. Restricted accuracy remains the behavioral primary endpoint. Raw probabilities are needed to detect improvement caused merely by changing the denominator of conditional probabilities.

Report the four S/G simple effects and their interaction:

`S|G=0 = y10-y00`, `S|G=1 = y11-y01`, `G|S=0 = y01-y00`, `G|S=1 = y11-y10`, and `I = y11-y10-y01+y00`.

For the eight-cell extension, report all conditional simple effects of S, A and D. Calculate contrasts per dictionary before averaging. Use one set of paired, query-stratified bootstrap draws for all cells and contrasts. Interaction is scale-dependent: report it on accuracy and raw target probability without calling a positive interaction proof of a unique synergistic circuit. Effects involving the guaranteed cell are useful diagnostics, not independent evidence that its full restoration was predicted empirically.

Recommended fresh-confirmation criteria for a **sufficient partial-restoration claim** are: native restricted accuracy ≥0.80 overall and ≥0.70 in each query stratum; primary candidate-minus-00 accuracy paired 95% lower bound >0.20; native-minus-candidate accuracy upper bound ≤0.05; and candidate-minus-a-prespecified relevant partial comparator lower bound >0.05. These thresholds are recommendations to fix before confirmation, not post-outcome defaults. If the candidate instead concerns a contrast without near-native restoration, formulate that narrower claim and its minimum effect explicitly; do not describe it as complete recovery.

Use 512 new confirmation dictionaries if feasible. A conjunction of the fixed main gates supports only that selected claim; unselected simple-effect intervals are descriptive unless multiplicity is addressed prospectively. Report all failures and individual intervals honestly, with no claim of simultaneous confidence coverage. Validity failure remains separate from scientific failure.

## Potential contribution and limits

A useful empirical result would be a prospective, substantial asymmetry between state correction, deletion of extra routes and restoration of missing routes, reproduced on fresh dictionaries and surviving the appropriate partial controls. It would identify a concrete boundary of E6's failed transfer in this checkpoint and task. Neither the full-restoration theorem nor merely running a larger model establishes novelty. Field-wide novelty requires comparison with prior activation-patching, attention-graph and context-restoration work once the candidate is specific.

Even a clear result concerns oracle interventions on one pretrained model and artificial serializations. It does not show that a deployable repair can discover pair relationships, that model scale caused the failure, or that all pretrained models use the same mechanism. The public E6 negative result remains part of the evidence regardless of E7's outcome.
