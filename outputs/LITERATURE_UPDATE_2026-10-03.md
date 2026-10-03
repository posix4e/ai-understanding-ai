# Recent papers and next experiments

Checked October 3, 2026. Targeted primary-source search, not an exhaustive claim
to have found every newest paper. This update was written after E4 confirmation;
it did not inform E4's frozen design. Earlier literature notes remain preserved.

| Paper and date | Finding | Authors' recommended direction |
| --- | --- | --- |
| [Are We Recovering Mechanisms?](https://arxiv.org/html/2610.02098v1), October 1, 2026 preprint | An intervention score can prefer a behaviorally worse circuit. Restoring intact signals corrected 96/100 selected ranking failures. | Report a fixed behavioral measure alongside interpretability scores, show the size of behavioral losses, and compare metrics and interventions. |
| [SAEScientist-Bench](https://arxiv.org/html/2609.09113v2), September 8; revised September 11, 2026; work-in-progress preprint | Agents identify concept-associated features better than they causally steer behavior. | Test multiple-feature circuits, open-ended hypotheses and more model families; strengthen human/judge validation and connect features to model editing. |
| [HyVE: Can Language Model Agents Be Helpful Circuit Explainers?](https://arxiv.org/html/2606.24026v2), June 23; revised September 2, 2026; accepted to Findings of EMNLP 2026 | Agents can explain supplied circuits, but incomplete tests and execution errors limit reliability. | Improve intervention helpers and constrained execution; test larger natural circuits with memorization safeguards; preserve failed experiments for review. |
| [CHIVE: Would This Change Your Answer?](https://arxiv.org/html/2608.16747v1), August 17, 2026 preprint | Tested activation tools did not improve counterfactual prediction over the transcript-only baseline in this setting. | Improve tools and elicitation, investigate rare behaviors with more sampling, and develop cleaner explanation evaluation. |

Date corrections to the earlier reading list: CHIVE's August 21 date belongs to
its [accompanying blog](https://alignment.anthropic.com/2026/chive/); the preprint
appeared August 17. HyVE now has a September 2 revision and an acceptance notice.

The October 1 paper is close conceptual precedent for restoring disrupted
computational context. Its recipient-activation patches repair circuit rankings;
E4's external edge mask repairs answers after serialization changes. That
distinction does not establish field-wide novelty. The paper also explicitly
does not establish minimal causal sets or superiority over random restoration.

Our next-step recommendation is an inference from these papers and E4, not the
authors' proposed experiment: freeze the visibility-repair rule, test fresh
layouts and newly trained models, and compare predictions with behavioral
baselines. Separately test individual-edge necessity and investigate E4's one
retained error. Successful repairs establish effects of interventions; they
do not uniquely identify a model's entire mechanism.
