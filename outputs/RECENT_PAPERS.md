# Recent papers relevant to AI understanding AI

Checked 2026-10-03 against author/institutional pages and arXiv. This is a targeted
reading list, not an exhaustive claim about the latest paper in the field.

| Paper | Date/status | Authors' direction and relevance |
|---|---|---|
| [SAEScientist-Bench](https://arxiv.org/abs/2609.09113) | Submitted September 8, revised September 11, 2026; work-in-progress preprint | Agents' feature selectivity outpaces their causal steering. Authors propose more model families, multi-feature circuits, open-ended hypotheses, and stronger judgment validation. |
| [Would This Change Your Answer? / CHIVE](https://alignment.anthropic.com/2026/chive/) | August 21, 2026 | Activation-reading tools did not improve counterfactual prediction over a transcript-only baseline in this evaluation. Authors emphasize direct interventional tests and distinguishing information visible behaviorally from extra evidence supplied by tools. |
| [Can Language Model Agents Be Helpful Circuit Explainers? / HyVE](https://arxiv.org/html/2606.24026v1) | June 23, 2026 preprint | Validation plans and execution remain failure points. Authors propose richer intervention helpers, constrained execution, larger natural circuits and connecting localization to explanation; warn about memorized published mechanisms. |
| [Addressing Divergent Representations from Causal Interventions](https://arxiv.org/html/2511.04638v5) | Originally November 2025; revised April 22, 2026 | Intervention-induced states can recruit unnatural pathways. Authors study a divergence-reducing loss and propose better identification/mitigation of harmful divergence; current mitigation is limited to simple settings. |
| [Formal Mechanistic Interpretability](https://arxiv.org/abs/2602.16823) | February 18, 2026; ICLR 2026 | Provides circuit robustness, patching and minimality guarantees, demonstrated on vision models. Transformer applicability requires further work. |

## Project-specific next steps — our interpretation

1. Compare explanations with stronger empirical and nonlinear baselines, not only
   additive effects or plausible prose.
2. Reserve multi-component interventions before discovery; examine redundancy,
   ablation and clean rescue separately from whole-path transplantation.
3. Compare zero, calibrated-mean and resampled corruptions and retain matched
   controls. Agreement is useful evidence, not a proof of on-distribution states.
4. Validate intervention code independently, including that unselected downstream
   components remain free to recompute.
5. Test variable layouts and new tasks before increasing scale or claiming general
   model understanding.

Experiment 2 addresses items 1–4 with six model seeds and 92 reserved composite
conditions. It does not implement formal verification or the divergence-loss
method, and does not reproduce the larger-model benchmarks above.
