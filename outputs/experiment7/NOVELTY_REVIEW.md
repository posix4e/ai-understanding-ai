# E7 prospective novelty review

**Checked 2026-10-04.** This is a targeted primary-source review for an experiment under discussion. No E7 model evaluation, input inspection, or outcome inspection was performed. E6 is disclosed discovery evidence, including its original failed validity classification; no E6 record is revised here. This document evaluates a possible contribution, not an observed E7 result or a claim of priority.

## Assessment

The broad ingredients are established: activation restoration, interactions between patches, positional steering, attention masks as computational graphs, and transformer permutation symmetry. A generic claim that “restoring context can repair behavior” or “early representations interact with later attention” would overlap closely with existing work.

The potentially useful empirical question is narrower:

> After a controlled serialization change and exact restoration of first-block value/query states, can a factorial intervention distinguish the contribution of the remaining early key-token states from the contribution of later attention visibility to a pretrained model's lookup failure, and can any proposed pattern survive separate prospective confirmation?

This is a candidate **diagnosis of one specific failure under explicit counterfactuals**, not a new general mechanism of binding. The informative outcomes are the two single-factor interventions, their conditional effects, and their interaction. Recovering native output after restoring both the complete native state and native graph is guaranteed by construction; that cell cannot be the novelty claim.

The reviewed papers do not supply this exact proposed protocol or predict its outcomes. A bounded search did not identify an exact match, but that does not establish absence from the literature. Confidence is high that the general methods are precedented, moderate that the proposed comparison has a distinct empirical target relative to the papers below, and low that its precise implementation is field-wide novel without further review and an informative result.

## Closest primary precedents

Each entry distinguishes what the source actually studies from this reviewer's comparison. Links identify the checked paper or version.

### 1. Intact computational-context restoration

Chuqin Geng et al., **Are We Recovering Mechanisms? Objective-Level Recovery Gaps in Mechanistic Interpretability**, arXiv:2610.02098v1, 1 October 2026. Section 5 replaces selected excluded incoming signals with recipient-intact activations while retaining the candidate circuits and their original behavioral rankings. It tests whether restoration corrects intervention-induced ranking errors. This directly precedes the general context-restoration rationale. E7 would instead manipulate an early residual-state factor and a later allowed-edge graph, measuring lookup outputs rather than circuit-ranking repair. [Primary paper, §5 and Appendix D](https://arxiv.org/html/2610.02098v1).

### 2. GPT-2 patching interactions

Sankaran Vaidyanathan, David Arbour, Aaron Mueller, Scott Niekum, and David Jensen, **The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching**, arXiv:2606.27510v1, 25 June 2026. The paper studies dependence of patch effects on other components, including pairwise/group interactions and GPT-2's indirect-object-identification circuit. This is strong precedent against claiming that factorial patching or non-additivity is itself novel. E7's factors would differ: whole key-token residual states versus the later visibility graph. Its difference-of-differences should be described as an intervention interaction, not automatically a unique path-specific causal effect. [Primary paper, §§3–5](https://arxiv.org/html/2606.27510v1).

### 3. Early-state and later-attention joint patching

Gregory Polyakov, Christian Hepting, Carsten Eickhoff, and Seyed Ali Bahrainian, **Interpretability Analysis of Arithmetic In-Context Learning in Large Language Models**, EMNLP 2025. Experiment 7 and Appendix D jointly patch an early MLP activation at an example-result position and later attention-module activations. This already investigates interactions between early information and its subsequent use. The proposed E7 comparison changes allowed attention edges rather than replacing later attention outputs, and uses exact first-block value restoration as its starting control. Those differences narrow the question; they do not erase the precedent. [Primary proceedings paper, Appendix D](https://aclanthology.org/2025.emnlp-main.92.pdf).

### 4. Associative recall and value-position restoration

Aryaman Arora et al., **Mechanistic evaluation of Transformers and state space models**, arXiv:2505.15105v3, 30 January 2026. The study corrupts association keys and restores representations at key, value, or query positions to distinguish association/retrieval mechanisms. It also studies a hierarchical retrieval task. This closely precedes causal localization of key–value associations at value positions. E7 cannot claim discovery of that general mechanism. Its candidate contribution concerns what remains causally unresolved after one subset of early states has already been restored under serialization change. [Primary paper, §§3–5](https://arxiv.org/html/2505.15105v3).

### 5. Positional steering in a compositional circuit

Cheng Tang, Brenden Lake, and Mehrdad Jazayeri, **Circuit explained: How does a transformer perform compositional generalization**, PLOS ONE 21(2):e0340088, 4 February 2026. Figure 12 swaps positional information in a circuit component while fixing the query pathway; attention and output logits shift toward a specified alternative. This directly precedes causal positional steering. E7 would preserve canonical token coordinates and vary states or graph connectivity, so its question is about residual failure under those supplied coordinates, not discovery that position can steer binding. [Primary article, Figure 12](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340088).

### 6. Binding states and position interventions

Jiahai Feng and Jacob Steinhardt, **How do Language Models Bind Entities in Context?**, arXiv:2310.17191v2. Section 3.2 and Appendix C intervene on apparent positions through rotary embeddings and assess binding representations. Appendix D.2 also uses grouped entity/attribute lists in grammatical parallel sentences. These precede separating binding content from position. They do not equate a position intervention with physically regrouping GPT-2 tokens while leaving a new causal mask in place. That distinction matters: representation movement, positional reassignment, and changed source availability are different interventions. [Primary paper, §3.2 and Appendices C/D.2](https://arxiv.org/html/2310.17191v2).

### 7. Causal visibility and position bias

Xinyi Wu, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie, **On the Emergence of Position Bias in Transformers**, arXiv:2502.01951v4, 9 August 2025. The paper models attention masks as directed graphs and analyzes their interaction with positional encodings across layers. Thus “the mask changes computation even when position information is supplied” is not a new conceptual claim. E7 would ask how much a particular failure depends on that graph after an explicitly defined early-state repair. [Primary paper](https://arxiv.org/html/2502.01951v4).

### 8. Separating sequence coordinates from attention order

Zhilin Yang et al., **XLNet: Generalized Autoregressive Pretraining for Language Understanding**, 2019. Section 2 distinguishes physical sequence order from factorization order: original positions are retained while attention masks implement different conditioning orders. This is longstanding precedent for separating these notions. XLNet trains with its permutation objective; it is not the proposed post-hoc GPT-2 diagnosis. Nevertheless, treating a permuted mask as an unprecedented way to represent an alternative order would be unjustified. [Primary paper, §2, page 3](https://arxiv.org/pdf/1906.08237).

### 9. Transformer permutation equivariance

Hengyuan Xu et al., **Permutation Equivariance of Transformers and Its Applications**, CVPR 2024. The work formalizes inter-/intra-token equivariance and discusses positional embeddings and masked attention. Its supplement restricts the fixed-mask decoder analysis to feature-column permutations; it must not be cited as proving arbitrary row permutation with an unchanged causal mask. The relevant E7 full-transport identity instead follows by permuting inputs, coordinates, and both mask axes consistently. This elementary coordinate change is an oracle check, not a discovery about trained weights. [Primary proceedings page](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Permutation_Equivariance_of_Transformers_and_Its_Applications_CVPR_2024_paper.html), [author-hosted supplement, §10.4](https://iqua.ece.utoronto.ca/papers/lxiang-cvpr24-supp.pdf).

### 10. Recent context-repair dynamics

Pranjal Garg and Jacob Beck, **Spontaneous Context Restoration: How Language Models Recover from Corrupted Inputs**, arXiv:2609.35475v1, 28 September 2026. The paper examines controlled transformers and pretrained models, including position-group activation patching and changes across layers. It is relevant precedent for separating early corruption effects from later propagation. Its spontaneous repair and token-corruption setting differs from E7's externally supplied state/graph interventions after token reordering. The terminology “context restoration” therefore carries no novelty by itself. [Primary paper, patching methods S1.11 and S2.6](https://arxiv.org/html/2609.35475v1).

### 11. Interpretation and metric cautions

Stefan Heimersheim and Neel Nanda, **How to use and interpret activation patching**, arXiv:2404.15255v1, 23 April 2024. This primary tutorial explains that patching design and outcome metrics constrain causal interpretation. It supports reporting the precise counterfactual and both restricted-choice and ordinary-generation outcomes. It does not prescribe E7's specific factors, sample, or success thresholds. [Primary paper](https://arxiv.org/abs/2404.15255v1).

## What the proposed factorial would identify

Let S0 be the full block-0 output under grouped canonical coordinates and the correct value guard. Let S1 replace only key-chunk residual states with their aligned native counterparts. If the registered restoration identities hold for values, prefix, and query suffix, S1 is the entire native block-0 state in grouped coordinates. Let G0 retain the grouped physical causal mask in blocks 1–11. Let G1 use the native mask transported to the grouped token indices in those blocks. Neither factor changes model weights.

For a prespecified output measure Y, estimate all four Y(Sa,Gb), the state effects under each graph, the graph effects under each state, and

\[
I=Y(S_1,G_1)-Y(S_1,G_0)-Y(S_0,G_1)+Y(S_0,G_0).
\]

“Separable” should mean independently manipulated experimental factors, not a presupposition that their effects are additive or that they correspond to two uniquely identified natural mechanisms. The magnitude and sign of I depend on the outcome scale. A nonlinear probability transform can create an interaction even when an underlying logit relation is simpler. Report the predeclared probability measure, accuracy, raw target probability, answer-token mass, and unrestricted generation; a logit-scale diagnostic can be declared in advance as a secondary check.

If H is the native residual array and P permutes token occurrences into grouped order, transporting a native additive mask M gives PMPᵀ. With aligned position coordinates and pointwise operations, the downstream computation obeys F(PH,PMPᵀ)=PF(H,M), up to numerical error. Hence Y(S1,G1) must reproduce native output. It checks implementation and identifies the intended coordinate mapping. Its guaranteed recovery is not empirical evidence for a newly discovered circuit.

The off-diagonal cells are informative. Recovery under S1,G0 would show sufficiency of early key-state restoration within this setting despite the later graph change. Recovery under S0,G1 would show sufficiency of later graph restoration despite the remaining initial key-state differences. Failure of both alone with valid joint recovery would show a joint requirement within this intervention family, not universal necessity or unique localization. Partial effects should remain quantitative; do not force a binary “which cause wins” account.

G1 can expose physically later context tokens. It must be labeled an oracle manipulation, not ordinary left-to-right inference or a deployable repair. It uses the known original ordering and pair structure. Native activation patches likewise supply counterfactual information from an intact execution. These limitations remain even if the single-factor cells perform well.

## What would make the empirical contribution more persuasive

These are recommendations from this review, **not instructions attributed to the cited authors**.

1. Freeze the four-cell comparison and its signed contrasts before new model outcomes. Use fresh dictionaries, retain all query strata, and preserve E6's negative/invalid records and any separately registered numerical repeat. Preregistration strengthens evidential credibility; it does not itself make a mechanism novel.
2. Treat the full-transport/full-state cell solely as a positive identity control. The proposed claim must rest on off-diagonal behavior or a prospective interaction, not guaranteed joint success.
3. Include a specificity check if claiming correct relational structure rather than generic donor-state or mask effects: for example, a prespecified misaligned-state donor or an appropriately degree-matched wrong transported graph. Define the control carefully before results. This is a proposal, not an assertion that any particular control is sufficient.
4. Preserve supplied coordinates and exact token occurrences, avoiding retokenization confounds. Report that grouped text is unusual; a result on this synthetic prompt is not ordinary-language order robustness.
5. If a pattern survives the first fresh sample, test a separately fixed prompt family or checkpoint before making a broader claim. More rows from one format reduce sampling uncertainty but do not establish cross-model generality.

An informative, replicated off-diagonal dissociation could support a narrow empirical contribution. A study showing only the guaranteed oracle identity, or only generic interaction with no controlled interpretation, would add little beyond the cited literature. The factorial is a justified next diagnostic, not a promise of a publishable discovery.

## Search record and unresolved coverage

Searches included combinations of GPT-2, associative recall, key/value reordering, activation patching, factorial interactions, context restoration, attention visibility, permutation masks, and transported attention masks. Primary arXiv versions, PLOS, ACL proceedings, CVPR proceedings, and an author-hosted supplement were checked. Recent leads were followed to their primary papers rather than cited from aggregators. The review checked relevant sections; it did not independently reproduce their experiments or certify every theoretical claim.

No reviewed source was found to report precisely this four-cell early-key-state versus transported-later-mask comparison after exact value-state restoration. This is a search observation, not evidence of field-wide absence. Terminology varies across causal tracing, interchange interventions, path patching, graph interventions, and order perturbations. Unindexed work, workshop papers, code-only experiments, and relevant appendices can remain. The safest current wording is **“we test this specific state–graph distinction prospectively”**, not **“we are the first to discover state–graph separation.”**

## Preforward amendment: eight-cell discovery grid

**Added 2026-10-04 after reading `experiment7/conditions.py` and `experiment7/DISCOVERY_PLAN.md`, before E7 outcomes.** The agreed design expands the four-cell comparison above into a complete 2×2×2 grid. E7 is explicitly exploratory: 128 fresh, excluded dictionaries and all eight cells are fixed before evaluation, but a promising pattern requires a separately registered confirmation on new inputs. The original four cells remain embedded in this larger grid. No further literature search was performed for this amendment.

Let P denote grouped physical visibility and T the transported native visibility. The later-block graph factor is now decomposed into deletion D and addition A:

| Later mask | D: remove P-only edges | A: add T-only edges |
| --- | --- | --- |
| P | 0 | 0 |
| P∩T | 1 | 0 |
| P∪T | 0 | 1 |
| T | 1 | 1 |

Each graph is crossed with no/yes native first-block key-state patching. Every factorial cell already receives the correct first-block value guard. D and A act in blocks 1–11; they do not alter that common first-block intervention. For the audited two-token chunks, deletion removes 24 future-logical-key→earlier-value token edges, while addition opens 24 earlier-value→later-key token edges per head in each intervened block. Equal edge counts do not mean equal removed attention mass, equal activation displacement, or equivalent receiving roles. The addition conditions remain oracle diagnostics because they open physically future visibility.

This design sharpens the possible empirical contribution to **an asymmetry between supplying missing later-stage context and removing extra later-stage context, conditional on restored first-block value/query states**. For example, if addition-only recovered behavior while deletion-only did not, the useful result would be sufficiency of the added connections in that declared state condition despite leaving the extra connections present. It would not establish that deletion is universally unnecessary, that each added edge is necessary, or that early key states have a context-independent role. The reverse pattern and absence of an asymmetry are equally legitimate outcomes.

Report addition and deletion effects within each early-state condition, their joint effect, and whether key-state patching changes those effects. The contrast Y(S,0,1)−Y(S,1,0) directly compares the two single graph changes; a full interaction decomposition is available from the eight cells. These remain scale-dependent exploratory contrasts, not a unique allocation of causal responsibility. Restricted-choice, raw probability, candidate-mass, and unrestricted-generation results must remain distinguishable.

None of the close precedents above makes a generic addition-versus-deletion comparison novel. The candidate distinction is the exact source/receiver classes, controlled starting states, and prospective behavioral asymmetry in this serialization failure. Both full transport from block 0 and native key-state patching plus transport in blocks 1–11 remain guaranteed equivalence controls. A compelling claim must arise from the non-guaranteed cells and survive new-input confirmation; the eight-cell expansion does not raise the current confidence in field-wide novelty by itself.
