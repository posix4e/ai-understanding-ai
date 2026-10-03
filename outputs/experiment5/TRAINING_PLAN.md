# Fixed fresh-model cohort for Experiment 5

Written before training or evaluation of this cohort. The E5 causal protocol
and prediction grid will be finalized separately before any shifted-layout
evaluation of these checkpoints. E3/E4 outcomes are disclosed discovery evidence.

Train exactly six new models, seeds 6, 7, 8, 9, 10 and 11, with the unchanged
root `train.py` and `model.py`: 2,000 AdamW steps; batch size 128; learning rate
.001; weight decay .01; gradient clipping 1; embedding initialization SD 1;
four heads, two layers, width 64 and MLP width 128; four key/value pairs and
16 distinct keys/values. Training batches use the existing seed+1000 schedule.
CPU only, four threads, deterministic PyTorch algorithms. Native validation
seed 100000 is reused as disclosed training monitoring, never confirmation.

Save every checkpoint, metadata and training log to
`outputs/experiment5/checkpoints/seed_SEED.*`. Do not replace seeds, select a
best checkpoint, extend training for low-performing members, or exclude failures.
The endpoint is the final step, regardless of validation performance. Training
failures stay in the cohort record. All training inputs will enter the later
holdout exclusion audit.

After training, freeze all six checkpoints with the complete E5 protocol and
inputs before any E5 shifted-layout model evaluation. The existing E1–E4
checkpoints and results will not be overwritten.

Command for each declared seed:

```sh
.venv/bin/python train.py --seed SEED --steps 2000 --threads 4 --embedding-std 1.0 --output-dir outputs/experiment5/checkpoints
```
