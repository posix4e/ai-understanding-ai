# Reproduce or audit the pretrained GPT-2 test

## Read the evidence

- [Protocol](../../experiment6/PROTOCOL.md) and [exact conditions](../../experiment6/conditions.py)
- [Calibration plan](../../experiment6/CALIBRATION_PLAN.md), [all calibration results](calibration_results.json) and [per-case calibration outputs](calibration_rows.csv)
- [Confirmation inputs](confirmation_inputs.json), [input audit](input_audit.json) and [token audit](confirmation_token_audit.json)
- [Blind predictions](../../experiment6/predictions.json) and [forecast outcomes](forecasts_vs_results.csv)
- [Raw results and batch ledgers](confirmatory), [all decoded predictions](all_predictions.csv) and [full results](RESULTS.md)
- [Independent numerical audit](independent_audit.json)

The original completed run's registration is
[`6ae8866e675cead908ae40c1862e81ee269eaf42`](https://github.com/posix4e/ai-understanding-ai/commit/6ae8866e675cead908ae40c1862e81ee269eaf42).
[Remote verification](remote_lock.json) preceded all reordered or repaired
pretrained-model forwards. The superseded registrations and first native-only
abort remain in the record. The [diagnostic](probability_diagnostic.json) explains
the float64 final softmax correction; model computation remains float32.

## Audit without running GPT-2

From the repository root, install the versions in
[requirements.txt](../../experiment6/requirements.txt) in a separate environment.
The commands below use `python` from that environment; the original local path
was `work/e6-venv/bin/python`.

The independent audit checks registered source and model files, token labels,
every batch hash, equality between batch and merged arrays, and independently
recomputed scores and intervals. It does not import the official scorer or run
a model. The local pinned model/tokenizer files are needed for hash checks, even
though their weights are never loaded. Full-state identities and runtime
parameter-preservation measurements remain runner evidence; saved final
probabilities cannot independently prove them.

Use a new audit output path. The current audit tool **refuses to overwrite any
existing record**, including the published original audit:

```sh
python -m experiment6.audit_outcomes --root outputs/experiment6 --protocol experiment6/protocol.json --manifest outputs/experiment6/confirmation_manifest.json --output work/e6-original-audit-check.json
```

The original audit was performed by an earlier version of that tool. Its exact
source bytes are preserved in [audit_original.py](audit_original.py), with SHA-256
`367465b65b87a08491a19bb06f587414b6955a9ef2401aac22befe6abe33e9d0`, matching
`audit_tool_sha256` in [independent_audit.json](independent_audit.json). The
snapshot was verified against that recorded hash when preserved. It is an
archival source copy, not the recommended executable entry point from its
archive location. The current tool adds configurable paths and overwrite
protection; do not replace the original record with a new audit.

To recompute the official scores without changing published files, copy the
saved arrays to a new scratch directory first:

```sh
set -e
mkdir work/e6-score-audit
cp -R outputs/experiment6/confirmatory work/e6-score-audit/confirmatory
python -m experiment6.score --root work/e6-score-audit --protocol experiment6/protocol.json
```

This produces scratch `scores.json` and `RESULTS.md`. It reads probabilities,
not model weights. Choose another unused scratch path if the directory already
exists.

## Run the actual model again

Create a separate environment and install `experiment6/requirements.txt`.
`python -m experiment6.download_model` downloads only the pinned public model and
tokenizer files into `work/models/gpt2`. The revision and all file SHA-256 hashes
are recorded in [model_manifest.json](model_manifest.json). No model weights are
trained, and no remote inference API is used.

Run:

```sh
set -e
python -m experiment6.run_phase --registration 6ae8866e675cead908ae40c1862e81ee269eaf42 --output-dir work/e6-repeat/confirmatory
python -m experiment6.score --root work/e6-repeat --protocol experiment6/protocol.json
python -m experiment6.audit_outcomes --root work/e6-repeat --protocol experiment6/protocol.json --manifest outputs/experiment6/confirmation_manifest.json --output work/e6-repeat-audit.json
```

Use a new empty output directory. The runner refuses to overwrite results or
silently restart a partial run. It checks frozen files against the registration
commit and verifies the model files before inference. Clone with the registration
commit available in local Git history. The published original outputs can be
compared with the new run without changing either set.

Execution used CPU, four threads, float32 weights/activations, float64 final
softmax, fixed full-vocabulary next-token argmax and 16-choice evaluation. See
[environment.json](environment.json) for exact package versions. The original
run used an isolated environment with the existing Torch installation visible
through a `.pth` file; a separate clean installation of the listed packages is
the portable equivalent. Repeated rows/conditions are not independent model
replications, and the result does not audit GPT-2's unavailable pretraining data.

## Reproduce the separately registered precision repeat

The precision repeat is registered at
[`6edaaac7c9a1eab41f4cc38ab700d212dcf866dc`](https://github.com/posix4e/ai-understanding-ai/commit/6edaaac7c9a1eab41f4cc38ab700d212dcf866dc).
See the [precision plan](../../experiment6/PRECISION_PLAN.md),
[precision configuration](../../experiment6/precision_protocol.json),
[precision manifest](precision_manifest.json), and
[remote verification](precision_repeat/remote_lock.json).

This is a disclosed numerical follow-up on the **same 512 dictionaries**, in
the same order and under the same fifteen conditions. It is not an independent
sample or a replacement for the original run. The same float32 checkpoint
parameter values are promoted exactly to float64. Computation and final
normalization use float64; the unchanged adapter's additive masks contain only
exact zero and negative infinity. The wrapper checks the original float32 value
hash, the promoted parameters before and after inference, and preservation of
the source values. All original state and behavioral thresholds remain fixed.

**Keep all published original files in place.** The precision manifest freezes
original-run evidence as well as code and inputs. Moving, deleting, regenerating,
or editing that evidence can fail registration verification. Use a checkout
containing the published originals and both registration commits. New runs must
write to a separate unused directory:

```sh
set -e
python -m experiment6.precision_repeat --registration 6edaaac7c9a1eab41f4cc38ab700d212dcf866dc --output-dir work/e6-fp64-repeat/confirmatory
python -m experiment6.score --root work/e6-fp64-repeat --protocol experiment6/precision_protocol.json
python -m experiment6.audit_outcomes --root work/e6-fp64-repeat --protocol experiment6/precision_protocol.json --manifest outputs/experiment6/precision_manifest.json --output work/e6-fp64-repeat-audit.json
```

The runner verifies the precision manifest against that public commit and the
checkpoint against its file hashes. The audit reads the completed run's recorded
registration automatically and refuses to run before `finished.json` and
`scores.json` exist. A failed or interrupted run must be retained; choose a new
directory for any explicitly documented repeat.

After the published precision run and its scoring are complete, its saved
outputs can be audited without another model forward:

```sh
python -m experiment6.audit_outcomes --root outputs/experiment6/precision_repeat --protocol experiment6/precision_protocol.json --manifest outputs/experiment6/precision_manifest.json --output work/e6-published-fp64-audit-check.json
```

The published precision repeat passed implementation checks and failed the
three primary repair requirements. See [the summary](SUMMARY.md).
The presentation exporter `python -m experiment6.export_precision` corrects
the unchanged scorer's generic report heading and its word “fresh” to identify
the reused sample. It also exports all 7,680 decoded predictions and the six
forecast comparisons. It does not change scores or original-run files.
`python -m experiment6.compare_precision` rebuilds the per-condition numerical
comparison from saved arrays. These presentation commands target the published
precision directory; use them only when deliberately rebuilding those reports.
