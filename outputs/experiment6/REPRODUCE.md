# Reproduce or audit the pretrained GPT-2 test

## Read the evidence

- [Protocol](../../experiment6/PROTOCOL.md) and [exact conditions](../../experiment6/conditions.py)
- [Calibration plan](../../experiment6/CALIBRATION_PLAN.md), [all calibration results](calibration_results.json) and [per-case calibration outputs](calibration_rows.csv)
- [Confirmation inputs](confirmation_inputs.json), [input audit](input_audit.json) and [token audit](confirmation_token_audit.json)
- [Blind predictions](../../experiment6/predictions.json) and [forecast outcomes](forecasts_vs_results.csv)
- [Raw results and batch ledgers](confirmatory), [all decoded predictions](all_predictions.csv) and [full results](RESULTS.md)
- [Independent numerical audit](independent_audit.json)

The final registration is
[`6ae8866e675cead908ae40c1862e81ee269eaf42`](https://github.com/posix4e/ai-understanding-ai/commit/6ae8866e675cead908ae40c1862e81ee269eaf42).
[Remote verification](remote_lock.json) preceded all reordered or repaired
pretrained-model forwards. The superseded registrations and first native-only
abort remain in the record. The [diagnostic](probability_diagnostic.json) explains
the float64 final softmax correction; model computation remains float32.

## Audit without running GPT-2

From the repository root, install the versions in
[requirements.txt](../../experiment6/requirements.txt). Recompute the official
scores with `python -m experiment6.score`. This reads saved probabilities only.
`python -m experiment6.export` rebuilds decoded prediction and forecast tables;
it needs the tokenizer files but does not run a model.

The independent `python -m experiment6.audit_outcomes` additionally checks the
registered model files and compares each batch with the merged arrays. It does
not import the official scorer or run a model. Full-state identities remain
runner measurements; saved final probabilities cannot independently prove them.

## Run the actual model again

Create a separate environment and install `experiment6/requirements.txt`.
`python -m experiment6.download_model` downloads only the pinned public model and
tokenizer files into `work/models/gpt2`. The revision and all file SHA-256 hashes
are recorded in [model_manifest.json](model_manifest.json). No model weights are
trained, and no remote inference API is used.

Run:

```sh
python -m experiment6.run_phase --registration 6ae8866e675cead908ae40c1862e81ee269eaf42 --output-dir work/e6-repeat/confirmatory
python -m experiment6.score --root work/e6-repeat
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
