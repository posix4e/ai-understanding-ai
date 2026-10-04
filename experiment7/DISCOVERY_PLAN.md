# E7 discovery plan: separate state damage from attention-path changes

Prepared 4 October 2026 before any pretrained E7 forward. This is an exploratory
study following E6, not a new confirmation. Every condition and outcome will be
retained. A promising empirical contrast must be tested with newly generated,
excluded-input confirmation and fixed rules before a discovery claim.

## Fixed setup

Use the same pinned GPT-2 small checkpoint, exact tokenizer, one_demo text format,
16 candidate answers, CPU four threads and float64 computation as E6's valid
precision repeat. No training, checkpoint search or prompt selection. Generate
128 dictionaries with seed10700001,32queries per pair,excluding all64E6 calibration
and512E6 confirmation association maps. Exclude identical maps regardless of pair
order or queried label. Native outputs are donors only for explicit patch cells.

The grouped order moves exact token occurrences and keeps their original position
IDs. Every factorial condition uses the E6 correct first-block value guard. This
restores native first-block value and suffix states within numerical precision;
key-chunk states can still differ. The two factors are:

1. Replace the first-block key-chunk residuals with the native residuals: no/yes.
2. For blocks1–11 use one of four complete attention masks:
   physical; physical∩transported-native (deletion only);
   physical∪transported-native (addition only); transported-native.

The transported-native mask permits edge(row,column) exactly when the column's
original token index is at most the row's original index. For this four-pair,
two-token-per-chunk layout, it removes24future-key→earlier-value token edges and
adds24earlier-value→later-key edges relative to grouped physical causality.
These changes affect allheads. Prefix,suffix,within-chunk andself edges stay fixed.

The additions can open edges to physically future tokens. They are explicitly
**oracle diagnostics**,not ordinary causal generation. The residual patches also
use a native-order donor. This experiment does not claim an autonomous repair.

The8factorial cells are accompanied by native,groupedcanonical,no-op,and full
transport in all12blocks. Full transport is a mathematical equivalence control.
Patching all damaged first-block keys and transporting all remaining layers is
also an equivalence control. Recovering those two controls is not a discovery.

## Validity and complete reporting

Before pretrained evaluation, validate the new arbitrary-mask adapter on random
weights, including agreement with unmodified GPT-2, oracle permutation equivalence,
patch no-ops, refusal of future edges without explicit authorization, and cleanup
of temporary hooks/buffers on failure. Verify pretrained model file hashes.
Use the same state tolerance1e-5 and full-probability tolerance1e-6 as E6. Check:
finite normalized outputs; unchanged parameters and buffers; native/guard adapter
agreement with frozen E6 on these same inputs; grouped no-op; guarded value and
query states; full-state equality after nativekeypatch; full-transport and combined
oracle output equality. Failed checks invalidate scientific interpretation.

Save all per-case16candidate probabilities and logits, unrestricted argmax token IDs, labels,
query strata, batch ledgers/hashes, and first-block state-error measurements.
Also save mean squared residual differences from native at each block for the
prefix,key,value andquery regions, without interpreting equal averages as equal
representations. All conditions use the same rows and shared bootstrap draws.

Report16-choice accuracy separately from full-vocabulary next-token accuracy,
raw target probability, conditional target probability andcandidate mass. Report
per-query results and paired descriptive intervals using2000stratified dictionary
bootstrap draws,seed10700003. No p-value search andno outcome-based exclusions.
The factorial contrasts and subsequent model/task scope remain exploratory until
separate confirmation. Any later exploratory extension must be documented before
its first forward and must not overwrite this grid.

## Novelty standard

A novel-discovery claim requires a non-guaranteed, specific empirical effect,
independent implementation checks, prospective replication on excluded inputs,
and a primary-source review distinguishing the claim from existing research.
Permutation equivalence, generic mediation interactions, a newly measured number,
or the mere failure of E6 is insufficient by itself. Field-wide novelty cannot
be proved by absence of a search result; unresolved close precedents prevent an
unqualified claim. Preserve contrary evidence and refine the explanation.

Pre-forward measurement addition: save candidate logits and report target logit
minus mean of the15other candidate logits as a secondary diagnostic. It is
independent of the final softmax normalizer and prevents interpreting a
probability-scale interaction as automatically a logit-scale interaction.
