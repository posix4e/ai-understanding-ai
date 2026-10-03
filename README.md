# AI understanding AI

An open, locally executed pilot: can an AI-written explanation of a tiny neural
network predict its behavior under causal interventions?

## Experiment 4: restoring key visibility repairs the failure

**A separately preregistered repair restored 99.95–100% lookup accuracy in all
six models, from 28.9–58.2% on the same fresh grouped inputs.** Keeping the
original position labels had not been enough. Blocking six newly exposed
key-attention edges per first-layer head restored the visibility each value had
during training, without retraining or copying native activations.

All registered gates passed on 2,048 new dictionaries per model, including
recovery in every query position and a mean target-probability advantage over
eight masks with matching edge counts. That advantage was 10.6–34.6 percentage
points across models. Some alternative masks nearly tied the repair, and one
guarded example still failed; this does not establish a unique or exact repair.

The nontrivial result is final-answer recovery while first-layer key states
remain altered. Restoration of the value states follows algebraically from the
intervention and is an implementation check. **176/180 numeric forecasts** met
the fixed .15 tolerance; the four misses underestimated two controls in one
model. E3's failed predictions remain unchanged below.

Read [the finding](outputs/FINDING.md), [complete E4 results](outputs/experiment4/RESULTS.md),
[independent review](outputs/experiment4/SKEPTICAL_REVIEW.md), and
[all forecast misses](outputs/experiment4/forecast_misses.csv).
The [public preregistration](https://github.com/posix4e/ai-understanding-ai/commit/b11245db8da95c4a98f9b25a452e828b7c3abe72)
and all 44 frozen file identities were verified at 14:45:51 UTC on 2026-10-03;
confirmation began at 14:45:59 UTC. This is a causal result in six small trained
models; field-wide novelty and transfer to other architectures remain unestablished.
See the [updated papers and next-step recommendations](outputs/LITERATURE_UPDATE_2026-10-03.md),
including the October 1 context-restoration paper found after E4 confirmation.

## Experiment 3: the strong position-binding explanation failed

**All six models failed the preregistered structural-transfer claim.** Grouping
the four keys before the four values reduced accuracy to 19–21%, despite 100%
accuracy on the original alternating layout. Restoring the original positional
labels raised accuracy only to 29–59%. Labels alone were not enough.

The test covered all 24 slot permutations, with matched coherent key/value
changes and value-only changes that predicted a particular wrong answer.
All **108 primary accuracy cells** missed their required thresholds, and
**937 of 1,200 numerical forecasts** missed the fixed .15 tolerance. The
implementation and no-op checks passed. Partial answer redirection occurred,
but it fell far short of the committed claim.

Read [the complete E3 results](outputs/experiment3/RESULTS.md),
[every forecast miss](outputs/experiment3/forecast_misses.csv), and
[the related-work audit](outputs/experiment3/RELATED_WORK.md). Prior work already
demonstrates positional steering; this is a prospective test of a stronger,
systematic prediction in our six models, not a claim of discovering that idea.

The failure motivated E4's separate test of changed causal visibility: grouped
values can attend to keys that were always in their future during training.
E4 confirmed a specified repair on fresh inputs; it does not rescue E3's claim.

## Experiment 2: head ablations and rescue

**All six models passed Experiment 2's registered prediction and baseline
comparison gates.** The test reserved 92 combinations of head ablations,
sole-head rescues and cross-layer interventions per model, across the original
three seeds and three new training seeds. No-op checks passed and clean held-out
lookup accuracy was 100% in every model.

The frozen explanation's family-balanced RMSE was **0.042–0.075** on mean target
probability, below the fixed 0.10 threshold. **502/552 novel forecasts** were
within 0.10; the per-model range was 78–88 out of 92, above the required 74.
It beat the strongest of seven frozen baselines in each model, with paired
95% loss-difference intervals below zero. Two baselines used matching per-case
discovery information. Controls do not contribute to the primary score.

This does not mean every forecast was right: **50 novel forecasts missed the
tolerance**, with a worst error of **25.3 percentage points**. The explanation is a discovery-fitted, second-order numerical model
of head interactions, not a uniquely identified internal algorithm. Its cross-layer
predictions equal one empirical baseline; added value there is not established.

Read [Experiment 2 results](outputs/experiment2/RESULTS.md),
[all forecasts and errors](outputs/experiment2/forecasts_vs_results.csv), and
[the independent skeptical review](outputs/experiment2/SKEPTICAL_REVIEW.md).

See the [new protocol](experiment2/PROTOCOL.md),
[prospective explanation](experiment2/PREDICTION.md), and
[recent papers and next-step rationale](outputs/RECENT_PAPERS.md).
The [preregistration commit](https://github.com/posix4e/ai-understanding-ai/commit/f4b3f1d73952e4fa449b2d4bf1f139769b0d75ad)
and all 62 frozen file identities were remotely verified at 14:00:33 UTC on
2026-10-03; confirmation started at 14:00:49 UTC. Experiment 1 remains preserved.

## Experiment 1: randomized dictionary lookup

A two-layer causal transformer receives four randomized key-value pairs followed
by a query key. It must output the associated value. The starting model has
70,720 parameters, four attention heads per layer, width 64, and MLP width 128.
Keys and values come from separate 16-token vocabularies and are sampled without
replacement within each dictionary. Query locations are uniformly randomized.

Two candidate accounts were **suggested in advance**, not independently discovered:
keys are written into subsequent value positions; or the query carries a positional
pointer to the answer. Key-swap and value-swap donors, plus targeted activation
patches, test these accounts. They need not be exhaustive or mutually exclusive.

## Result: a narrow compositional prediction succeeded

All three models achieved **100% accuracy on 2,048 held-out lookup inputs**.
All **159 frozen mean-effect forecasts** (53 interventions per model) passed the
fixed absolute-error tolerance 0.15. The largest mean forecast error was **0.00269**.
Results favor key-dependent routing information at value positions: L2 key
patches redirect lookup, value patches transfer content, and query patches have
near-zero effects. This does not uniquely identify the represented algorithm.

The strongest unseen intervention combined keys from a key-swap donor with
values from a value-swap donor. The explanation predicted cancellation back to
the original answer: this occurred in **100%, 99.95%, and 99.95%** of cases.
A simple additive baseline predicted the opposite effect. This one intervention
drives the overall forecast advantage; the empirical discovery baseline was
slightly better on familiar interventions. Individual patch failures are retained.

Read the **[results and limitations](outputs/RESULTS.md)**,
[independent skeptical review](outputs/SKEPTICAL_REVIEW.md), and
[complete forecast table](outputs/forecasts_vs_results.csv).

## Registration and failures

Initial runs achieved about 24–25% accuracy; longer training and higher learning
rates also failed. Increasing embedding initialization SD recovered the task.
All attempts and checkpoints are preserved in the
[training recovery record](outputs/training_recovery.md).

The first registered test aborted before any model evaluation because exact
holdout auditing found four training-overlap incidences. The failure is preserved
in [the original audit](outputs/confirmatory/holdout_audit.json). A disclosed v2
revision froze 2,048 recipient/donor bundles after deterministic exact-overlap
filtering (four rejected out of 2,052 candidates). Forecasts remained byte-identical.

The [v2 preregistration](https://github.com/posix4e/ai-understanding-ai/commit/b2d3bdd55db2dd9a8f2f3f2548dabe9fe8f53b37)
was read back from GitHub, checking all 45 frozen file identities at 12:45:04 UTC
on 2026-10-03. Confirmation started at 12:48:07 UTC. See the
[remote verification](outputs/remote_lock_v2.json), [protocol](PROTOCOL.md),
[explanation](PREDICTION.md), [numeric forecasts](predictions.json), and
[frozen hashes](preregistration_manifest_v2.json).

## Execution and roles

All training and experimental computation runs on an Apple M3 MacBook Air with
16 GiB RAM, using CPU PyTorch. No cloud compute or paid inference API is used for
experiments. AI research roles run in the existing Codex session (Astra requested
for implementer, blinded predictor, and skeptical reviewer); these conversational
roles are not locally hosted model inference. Blinding is procedural, not an OS
security boundary: roles shared a filesystem, and confirmation outputs did not
exist until forecasts were frozen. Only concise scientific outputs are published.

See [machine metadata](outputs/machine.json), [lab log](LAB_LOG.md), and
[dependencies](requirements.txt). Checkpoints and raw results are versioned.

## Reproduction

```sh
uv venv .venv --python python3.14
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/preflight.py
.venv/bin/python train.py --seed 0 --steps 2000 --threads 4 --embedding-std 1.0 --output-dir outputs/retrained
```

Repeat with seeds 1 and 2. The new output directory preserves published checkpoints.
Commands for every exploratory attempt are in the training record. To rerun the
exact frozen evaluation, preserve the published results first:

```sh
mkdir -p work
mv outputs/confirmatory_v2 work/published-confirmatory-v2
.venv/bin/python evaluate.py --phase confirmatory
.venv/bin/python score.py
.venv/bin/python build_report.py
```

Use a fresh checkout or a new archive directory for each rerun. The evaluator
requires GitHub CLI access for a read-only remote check, verifies frozen hashes,
repeats the exact overlap audit, and refuses to overwrite existing results.
Scoring and reporting consume saved arrays only. All
[checkpoints](outputs/trained), [frozen inputs](outputs/holdout_v2.npz),
[raw outcomes and scores](outputs/confirmatory_v2) are included.

## Claim limits

Explaining one trained model does not establish how learning works or explain
frontier AI. Six training seeds remain a small pilot. Causal transplantation
can establish effects of specified interventions without uniquely identifying the
represented algorithm. Exploratory findings and confirmatory results are kept
separate; failures and deviations remain visible.

Experiments 3 and 4 tested one changed layout and a specified repair. Next: test
the repair across held-out layouts and newly trained models, isolate which
edges are necessary, and investigate the retained failure case. Comparisons
across position schemes and intervention-induced states would test broader
mechanistic claims. These further experiments have not run.
