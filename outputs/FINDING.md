# A causal repair of failed dictionary lookup

**Restoring training-time key visibility recovered 99.95–100% accuracy in six
small transformers, without retraining.** The same models achieved only
28.9–58.2% on the fresh grouped inputs before the repair.

The models learned to read four alternating key/value pairs. Moving all keys
before all values disrupted lookup. Our first structural prediction failed:
preserving the original position labels did not preserve behavior. Experiment 3
failed all its primary accuracy gates, with 937/1,200 numerical forecast misses.

Experiment 4 tested a new explanation on 2,048 fresh dictionaries: grouping lets
a value attend to keys that were hidden from it in the training order. We
removed those six newly exposed links in every first-layer attention head. The
mask uses the known training layout; it never uses the correct answer or copies
activations from a successful run.

| Training seed | Grouped accuracy | Repaired accuracy |
| --- | ---: | ---: |
| 0 | 53.32% | 100% |
| 1 | 56.45% | 100% |
| 2 | 28.91% | 100% |
| 3 | 38.53% | 100% |
| 4 | 34.08% | 100% |
| 5 | 58.20% | 99.95% |

All six models passed every registered gate. The repair exceeded 95% accuracy
at every query position and improved correct-answer probability over the mean
of eight masks removing the same numbers of edges. The probability advantage
was 10.6–34.6 percentage points; every paired 95% lower bound exceeded the
registered five-point requirement. These intervals condition on the six fixed
checkpoints and sampled dictionaries, not a population of new trained models.

The result is near-complete behavioral recovery while first-layer key states
remain changed. Exact restoration of value states follows from the intervention
mathematically; it is not itself the empirical finding. Several alternative
masks also work well, including two that nearly tie the repair in one model.
One guarded example remains wrong and is fixed by additionally restoring key
states. The evidence does not establish a unique repair or necessity of each edge.

The explanation predicted 176/180 numerical outcomes within the committed .15
tolerance. The four misses underestimate two alternative masks in seed 3.
The controls match edge counts, not removed attention mass or activation norms.
All six models share a tiny two-layer architecture and fixed four-pair task;
arbitrary-layout transfer, other position schemes, and field-wide novelty remain
untested or unestablished.

Code, inputs, checkpoints, forecasts and scoring rules were
[publicly frozen before evaluation](https://github.com/posix4e/ai-understanding-ai/commit/b11245db8da95c4a98f9b25a452e828b7c3abe72).
All 44 identities were remotely verified at 14:45:51 UTC on October 3, 2026;
confirmation began at 14:45:59 UTC. An independent skeptical role checked all
90 raw condition/model cells, all forecast errors, 450 accuracy intervals, and
18 bootstrap comparisons. The roles share a filesystem; blinding was procedural.

See [full results](experiment4/RESULTS.md),
[independent review](experiment4/SKEPTICAL_REVIEW.md),
[retained forecast misses](experiment4/forecast_misses.csv), and
[related-work audit](experiment3/RELATED_WORK.md). A subsequent
[literature update](LITERATURE_UPDATE_2026-10-03.md) adds an October 1 paper on
repairing circuit rankings by restoring computational context, a relevant
conceptual precedent. All model training and
experimental evaluation ran locally on an M3 MacBook Air.

The next informative test is to freeze this repair rule and evaluate it across
new layouts and independently trained models. Separately removing or restoring
individual edges can test necessity. The single retained failure provides a
concrete case for investigating when changed key states still matter.
