# Scientific manuscript and overview review

Reviewed `paper/assemble.py`, generated `manuscript.json`, and the standalone plain-language overview against the independent E5 raw-data audit. This is an internal scientific review, not external peer review. No frozen source, model, or outcome file was changed.

## Assessment

The main claim is supported at the stated scope: prospective transfer of an externally supplied visibility repair across six fresh tiny models and the complete 105-layout class. The manuscript distinguishes algebraic state identities from empirical answer recovery, exposes the edge-magnitude failure, and limits broader boundary accuracy to the tested 256-dictionary subset. It correctly states that own-key/self states are generally not native states. It does not claim behavioral success for own-key/self on all 2,520 layouts.

The draft is suitable for responsible-author review and development as a reproducible synthetic-mechanistic case study. This does not establish field-wide novelty, independent replication, or readiness for a particular venue's submission requirements. The strongest contribution is the prospective test across an exhaustive finite intervention/layout class, with a falsifiable theory and retained counterexamples. The elementary restoration identity and position/context dependence themselves have close prior work.

## Numerical verification

The following agree with the audited evidence: six seeds; 105 primary and 2,415 boundary layouts; 1,044 primary condition cells per model; 624 alternative layout/mask cells across 102 eligible layouts; 15 primary errors in seed 7 involving two dictionaries and 15 layouts; worst primary layout/query accuracy 255/256; mean-control gains of 4.60–25.94 percentage points; failed edge thresholds in seeds 7 and 10; and 65/72 aggregate forecasts within .05.

All 108 numerical values and interval endpoints underlying the three figures exactly match the saved scores after percentage scaling. Primary table quantities and units agree. The 94.44-second training statement is the sum of in-process checkpoint timers (94.4382 seconds), not the full cohort launcher wall time. The confirmation duration is 23.43 minutes. Raw coverage of 492 shards and 21,301,248 repeated condition-row observations is correct; the manuscript appropriately does not treat these as independent examples.

The figure caption correctly cautions that zero-error bootstrap intervals are degenerate and do not imply zero population risk. Boundary means are scoped to each policy, and neither boundary success nor numerical forecast coverage is used to erase a failed claim.

## Corrections identified and checked

The regenerated manuscript corrects two scientific wording problems found in the initial source:

1. The restoration-class characterization now concerns **every value simultaneously**. An individual value can retain all its native predecessors in a boundary layout, so the initial singular formulation was false.
2. The statement that all boundary layout/query cells pass .95 now explicitly refers to **logical-prefix and own-key/self**. It is false for unguarded and some keyguard cells.

The overview now says that the required minimum effect was not shown with the specified confidence in two models. This is more accurate than saying both point effects were too small: seed 10's mean is .05313, above .05, but its lower confidence bound is .04500. The tested endpoint is an average across six individually re-added edges; the failed criterion was not imposed separately on every edge.

The overview also now describes **624 comparisons with alternative connection patterns across the input orders**, correcting the initial suggestion of 624 distinct connection rules. Patterns can recur across layouts. This affects the description of the control count, not any experiment or score. All necessary corrections found in this review are resolved in the current text.

## Citation checks

The publisher identifies the first author as **Cheng Tang** and supplies the citation **Tang C, Lake B, Jazayeri M**. The manuscript's initial “Tang S” has been corrected in the generated record. The title and publication date, 4 February 2026, agree with the [PLOS article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340088).

Author lists, versions, dates and the limited descriptions in the draft also agree with the primary records for [Arora et al.](https://arxiv.org/abs/2505.15105), [Wu et al.](https://arxiv.org/abs/2502.01951), and [Geng et al.](https://arxiv.org/abs/2610.02098). The draft appropriately treats these as substantive precedents rather than claiming priority over positional binding or context restoration.

## Scientific emphasis to retain

The seed-7 errors are rare but can be severe: one correct-answer probability falls from .999083 natively to .008674 after the partial guard, despite restored value states. This supports the manuscript's distinction between first-layer restoration and exact output recovery. Including one such concrete counterexample would strengthen the account, although the current draft does retain the errors and oracle recovery.

The plain-language explanation is otherwise faithful. It says the repair receives known pair relationships and original position labels, limits conclusions to small four-pair models, keeps the failed prediction visible, and makes publication a defensible possibility rather than a guarantee. Further experiments are proposals, not prerequisites invented to invalidate the measured result.
