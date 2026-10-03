# AI understanding AI

An open, locally executed pilot: can an AI-written explanation of a tiny neural
network predict its behavior under causal interventions?

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

## Status

Discovery complete; confirmation awaits preregistration. Three initial 2,000-step
runs achieved about 24–25% accuracy; longer training and higher learning rates
also failed. Increasing embedding initialization SD recovered the task. The
three final models achieved 100% accuracy on 1,024 discovery inputs. All failed
runs are preserved. See [training recovery](outputs/training_recovery.md).

Key-swap discovery patches favor routing through keys at value positions; query
patches have near-zero effects. A separate prediction role sees discovery data
only and will freeze numerical forecasts for held-out inputs and previously
untested joint patches. Its explanation, protocol, executable code, checkpoints,
and hashes will be committed and read back from GitHub before confirmation.

## Execution and roles

All training and experimental computation runs on an Apple M3 MacBook Air with
16 GiB RAM, using CPU PyTorch. No cloud compute or paid inference API is used for
experiments. AI research roles run in the existing Codex session (Astra requested
for implementer, blinded predictor, and skeptical reviewer); these conversational
roles are not locally hosted model inference. Blinding is procedural, not an OS
security boundary: roles share a filesystem, and confirmation outputs will not
exist until the forecast is frozen. Only concise scientific outputs are published.

See [machine metadata](outputs/machine.json), [lab log](LAB_LOG.md), and
[dependencies](requirements.txt). Checkpoints and raw results are versioned.

## Reproduction

```sh
uv venv .venv --python python3.14
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python train.py --seed 0 --steps 2000 --threads 4 --output-dir outputs/checkpoints
```

Repeat with seeds 1 and 2 for the initial pilot. Final commands and preregistration
will be recorded as the pilot matures. Do not overwrite archived runs.

## Claim limits

Explaining one trained model does not establish how learning works or explain
frontier AI. Three training seeds remain a small pilot. Causal transplantation
can establish effects of specified interventions without uniquely identifying the
represented algorithm. Exploratory findings and confirmatory results are kept
separate; failures and deviations remain visible.
