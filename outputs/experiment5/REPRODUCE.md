# Reproduce or audit Experiment 5

This guide documents the frozen entrypoints and artifact checks. It was drafted while the registered evaluation was running; it contains no numerical scientific findings. It is explanatory documentation, not an amendment to the protocol or predictions.

The public scientific freeze is [de745644b5c71853755e8d53cc9b3dec2abdc2ed](https://github.com/posix4e/ai-understanding-ai/commit/de745644b5c71853755e8d53cc9b3dec2abdc2ed). The original [remote lock](remote_lock.json) records verification at **2026-10-03T16:46:17.917703+00:00**. There are **95 manifest entries plus the manifest itself: 96 verified Git blobs**. The immutable files include scientific code, conditions, forecasts, protocol, inputs, six trained checkpoints, and supporting historical/audit artifacts. Raw E5 outcomes are subsequent artifacts, checked against their completion ledgers rather than included retrospectively in that freeze.

## Environment

Run commands from the repository root. The recorded host is Apple M3, 16 GiB RAM, macOS 27.0.1 arm64, Python **3.14.8**, PyTorch **2.14.1**, and NumPy **2.5.3**. All model work uses CPU, four PyTorch threads, one interop thread, and deterministic algorithms. No GPU, hosted model, or inference API is required. The frozen root [requirements.txt](../../requirements.txt) pins the complete Python environment; `uv` and GitHub CLI `gh` are external command-line tools, not Python dependencies.

In a separate checkout, with Python 3.14.8 available:

```sh
uv venv .venv --python 3.14.8
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python -m unittest experiment5.test_runner experiment5.test_score -v
```

The tests use synthetic arrays and randomly initialized models, never the trained checkpoints or confirmation inputs. Scientific evaluation and scoring use the frozen files, not regenerated conditions or a changed dependency version. Different hardware or package builds can change floating-point results even under deterministic settings; exact artifact identity and numerical reproduction are distinct checks.

## Audit the published package without running models

Use a checkout containing the published outcome package, its original `remote_lock.json`, and the frozen commit in local Git history. This command verifies local SHA-256 hashes, the lock's exact file list, and each file against the pinned Git object. It neither executes models nor changes files:

```sh
.venv/bin/python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
commit = 'de745644b5c71853755e8d53cc9b3dec2abdc2ed'
lock = json.loads(Path('outputs/experiment5/remote_lock.json').read_text())
manifest_path = Path('experiment5/manifest.json')
expected = json.loads(manifest_path.read_text())['files']
expected[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert lock['commit'] == commit and lock['files'] == expected
assert len(expected) == lock['verified_blob_count'] == 96
for name, digest in expected.items():
    data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, name
    assert subprocess.check_output(['git', 'show', f'{commit}:{name}']) == data, name
print('All 96 frozen files match their SHA-256 values and pinned Git blobs.')
PY
```

To also check that the pinned commit remains available from GitHub, the existing-lock API is:

```sh
.venv/bin/python -c 'from experiment5.preregister import verify_local_lock; lock = verify_local_lock(); print(lock["commit"], lock["verified_blob_count"])'
```

This uses `gh api` and therefore needs working GitHub CLI access. It validates every local locked hash and the remote commit identity. Do **not** call `--prepare` to manufacture a replacement registration. The CLI `--verify COMMIT` is intentionally restricted to a fresh checkout without an existing lock or confirmation directory; it is not the post-publication audit command.

## Audit raw shards and reproduce scoring

Wait for `outputs/experiment5/confirmatory/finish.json`. Preserve the published result package before regenerating derived tables, because the scorer and report overwrite their derived outputs. A separate clone or copied package is the simplest working location; keep the original raw `confirmatory/` directory and original `remote_lock.json` unchanged.

```sh
.venv/bin/python -m experiment5.scorer
.venv/bin/python -m experiment5.report
```

These two commands perform **no model forwards**. The scorer checks the completion seed/condition counts, each indexed shard's SHA-256, path and layout coverage, every expected condition/metric, finite arrays, normalized 16-class probabilities, argmax/accuracy/target consistency, and native-probability diagnostics. State-error values are saved runner measurements; their meaning is additionally tested by the frozen random-model suite.

Each seed has `confirmatory/seed_SEED/index.json`. Its `panels` dictionary contains `references`, `primary`, `edge_panel`, and `boundary`; each panel has a `shards` list. An entry records `path`, `sha256`, `layout_ids`, `n`, `condition_count`, `completed_at_utc`, and `elapsed_seconds`. Array keys inside each NPZ are exactly:

```text
layout_ID/condition_ID/metric
```

The scorer uses the fixed 2,000 paired, query-stratified bootstrap draws: seed 10200001 for primary/edge and 10200002 for boundary. It reuses weights across all models and conditions within the panel. Boundary uses the first 256 balanced primary dictionaries, not a separate independent dataset.

## Re-run the frozen model evaluation safely

Create a **new directory**, leaving the published outcome package and original lock untouched. The registration commit already contains the required inputs and checkpoint bytes. Checking out this earlier commit avoids copying completed E5 outcomes into the new evaluation directory:

```sh
git clone --no-checkout https://github.com/posix4e/ai-understanding-ai.git ai-understanding-ai-e5-reproduction
cd ai-understanding-ai-e5-reproduction
git checkout --detach de745644b5c71853755e8d53cc9b3dec2abdc2ed
uv venv .venv --python 3.14.8
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python -m experiment5.preregister --verify de745644b5c71853755e8d53cc9b3dec2abdc2ed
.venv/bin/python -m experiment5.run_phase
.venv/bin/python -m experiment5.scorer
.venv/bin/python -m experiment5.report
```

The fresh verification writes a **new reproduction lock with its current verification time**, referring to the same known frozen commit and file hashes. It does not replace or backdate the original publication lock. Keep the checkpoints and inputs unchanged. Do not retrain into `outputs/experiment5/checkpoints`, regenerate inputs over `inputs.npz`, or alter the protocol to make the verifier accept different files.

The controller refuses to overwrite a completed confirmation. If the new run is interrupted, use:

```sh
.venv/bin/python -m experiment5.run_phase --resume
```

Resume requires the same lock/checkpoint identities and verifies existing indexed shards before reuse. An unindexed shard or hash mismatch causes an error for explicit integrity review; do not delete the inconvenient file and silently replace it. `start.json`, optional `resume_events.jsonl`, per-seed ledgers, and `finish.json` document execution. Comparing a reproduction to published raw arrays should distinguish byte identity from numerical agreement: elapsed times and other provenance fields are expected to differ.

## Training reproduction is a separate check

The six seeds are exactly **6, 7, 8, 9, 10, 11**. The public [training plan](TRAINING_PLAN.md) fixed the final 2,000-step checkpoint, batch 128, AdamW learning rate .001, weight decay .01, gradient clipping 1, embedding standard deviation 1, four threads, and training generator seed `model_seed + 1000`. Native validation seed 100000 was repeated training monitoring, never confirmation. Every model has two layers, four heads, width 64, MLP width 128, and 70,720 parameters.

For an optional independent training check, write to an unused scratch directory and retain the registered checkpoints:

```sh
for seed in 6 7 8 9 10 11; do
  .venv/bin/python train.py --seed "$seed" --steps 2000 --threads 4 --embedding-std 1.0 --output-dir work/e5-training-reproduction
done
```

The training script itself permits overwriting an existing output stem, so use a new empty destination. Do not substitute retrained files into the locked evaluation. Timing metadata and serialization can prevent whole-checkpoint byte equality even when tensor values agree; compare `state_dict` tensors separately if auditing training determinism. Changing the cohort, initialization, steps, layout training, or sample selection constitutes a new experiment.

The frozen `experiment5.data` script documents deterministic input generation and historical exclusions. Reproduction of the registered evaluation uses its **saved** input file. Its generation command refuses to replace an existing file and needs historical inputs plus all six training metadata records; re-running it belongs in a separate generation-audit workspace.

## Compute and storage inventory

| Item | Registered workload or measured file size |
|---|---:|
| Fresh trained checkpoints | 6 × 70,720 parameters; 1,754,684 bytes total `.pt` files |
| Training elapsed time | About 94.44 seconds summed over six recorded runs, including their validation |
| Registered compressed input file | 14,702 bytes; 1,024 dictionaries |
| Condition specification JSON | 5,706,046 bytes |
| Primary conditions | 1,044 per model × 1,024 cases |
| Boundary conditions | 9,660 per model × 256 cases |
| Edge / reference conditions | 6 / 2 per model × 1,024 cases |
| Total saved condition/model cells | 64,272 |
| Total saved case-condition rows | 21,301,248 |
| Expected raw shards | 82 per model; 492 total |
| Uncompressed numerical array payload | Approximately 2.407 GB, excluding archive headers, ledgers, and CSVs |

Each case-condition row stores 16 float32 class probabilities, float32 target probability, int64 argmax, int8 accuracy, and nine float32 error diagnostics: 113 bytes of numerical payload. Shards contain at most 32 layouts. The largest possible primary shard has approximately 44.4 MB of array payload before archive overhead, and the controller enforces a compressed file size below 95 MB. No full activation caches are saved. Streaming one shard at a time keeps memory practical on the recorded 16 GiB host. Dependency environments, Git history, copied audit packages, and large CSV tables need additional disk space; budget several GB beyond the raw numerical payload.

The inventory's evaluation workload counts saved endpoints, **not** the number of internal model calls: the runner also computes native/layout/reference caches and oracle checks. Final evaluation time is obtained from completed ledger entries rather than estimated from the training time. After completion, this read-only inventory command totals actual compressed bytes and recorded shard durations:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/experiment5/confirmatory')
assert (root / 'finish.json').is_file(), 'Evaluation is not complete'
entries = []
for seed in range(6, 12):
    ledger = json.loads((root / f'seed_{seed}/index.json').read_text())
    entries.extend(entry for panel in ledger['panels'].values() for entry in panel['shards'])
assert len(entries) == 492
print({'shards': len(entries),
       'compressed_bytes': sum(Path(e['path']).stat().st_size for e in entries),
       'summed_shard_seconds': sum(e['elapsed_seconds'] for e in entries)})
PY
```

Summed shard times include per-shard evaluation and writing but exclude setup and scoring. `start.json` to `finish.json` gives evaluation wall-clock duration, including interruptions if any; neither figure includes all earlier exploratory work.

## Output roles

* `confirmatory/seed_*/{panel}/shard_*.npz`: full per-case numerical evidence; keep immutable.
* `confirmatory/seed_*/index.json`, `start.json`, `finish.json`: hashes, coverage, timing, and run provenance.
* `confirmatory/scores.json`, `global_decisions.json`, `forecast_statistics.json`: frozen-rule scoring and separate scientific/forecast decisions.
* `condition_results.csv`, `per_query_results.csv`: all conditions, individual Wilson intervals, displayed-value probabilities, and missing-context metadata.
* `controls_summary.csv`: within-layout mean controls and descriptive best controls.
* `primary_failures.csv`: every primary layout/query cell below its registered threshold.
* `primary_failure_cases.csv`: every wrong correct-guard prediction, even inside passing cells, with native probability and key-state error.
* `forecasts_vs_results.csv`, `forecast_misses.csv`: every one of 72 aggregate forecasts and every miss.
* `RESULTS.md`: readable report; `input_audit.json`, `training_cohort.json`, `THEORY.md`, and review documents record design context and limits.

Empty failure CSVs retain headers. No failed seed, layout, case, or forecast is removed from the prescribed reporting.
