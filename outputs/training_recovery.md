# Training recovery and model selection

The final models are `trained/seed_0.pt`, `trained/seed_1.pt`, and `trained/seed_2.pt`. Each attained 100% accuracy on the same 5,120-example **pilot validation** set after 2,000 training steps. This set was reused during training and model selection; these figures are not confirmatory results.

All runs use the same randomly sampled four-pair dictionary task, 16 distinct key token types, 16 distinct value token types, two transformer layers, four heads, width 64, MLP width 128, pre-layer normalization, and learned positions. Each model has 70,720 parameters. Training uses fresh batches of 128, AdamW, weight decay 0.01, gradient clipping at 1.0, four CPU threads, and deterministic PyTorch execution. Model seeds are 0–2, corresponding training-data seeds are 1000–1002, and pilot-validation seed is 100000.

| Artifact directory | Seed(s) | Steps | Learning rate | Embedding initialization SD | Final pilot accuracy |
|---|---|---:|---:|---:|---|
| `checkpoints` | 0, 1, 2 | 2,000 | 0.001 | 0.02 | 24.04%, 24.61%, 24.04% |
| `pilot_long` | 0 | 12,000 | 0.001 | 0.02 | 24.80% |
| `pilot_lr003` | 0 | 12,000 | 0.003 | 0.02 | 25.51% |
| `pilot_lr01` | 0 | 12,000 | 0.01 | 0.02 | 25.59% |
| `pilot_embed1` | 0 | 12,000 | 0.001 | 1.0 | 100% |
| `trained` | 0, 1, 2 | 2,000 | 0.001 | 1.0 | 100%, 100%, 100% |

The first models plateaued near the 25% accuracy attainable by choosing among the four displayed values without resolving the queried key. Longer training and higher learning rates did not recover the task. Increasing the initialization SD of both token and position embeddings to 1.0, while retaining linear-projection initialization SD 0.02, recovered performance: seed 0 reached 100% pilot accuracy at 1,000 steps. The extended successful run remained at 100% through 12,000 steps. This supports selecting the larger embedding initialization for this experiment; it does not by itself establish why it works.

We selected 2,000 steps for the final runs after observing the successful pilot. All three seeds completed this fixed schedule; none was selected or discarded based on downstream intervention results. The final validation losses were 0.00071505, 0.00068060, and 0.00063765 for seeds 0, 1, and 2. The original failed runs and every exploratory run remain preserved with `.pt`, `.csv`, and `.json` artifacts. Older checkpoints omit `embedding_std`; loading them defaults to their original SD 0.02. New checkpoints explicitly store the initialization setting.

## Commands

Run from the repository root. The commands reproduce each run but will replace identically named artifacts; choose a different output directory when preserving existing results.

```sh
for seed in 0 1 2; do
  .venv/bin/python train.py --seed "$seed" --steps 2000 --threads 4 --output-dir outputs/checkpoints
done
.venv/bin/python train.py --seed 0 --steps 12000 --threads 4 --eval-every 500 --output-dir outputs/pilot_long
.venv/bin/python train.py --seed 0 --steps 12000 --threads 4 --lr 0.003 --eval-every 1000 --output-dir outputs/pilot_lr003
.venv/bin/python train.py --seed 0 --steps 12000 --threads 4 --lr 0.01 --eval-every 1000 --output-dir outputs/pilot_lr01
.venv/bin/python train.py --seed 0 --steps 12000 --threads 4 --lr 0.001 --embedding-std 1.0 --eval-every 1000 --output-dir outputs/pilot_embed1
for seed in 0 1 2; do
  .venv/bin/python train.py --seed "$seed" --steps 2000 --threads 4 --lr 0.001 --embedding-std 1.0 --eval-every 250 --output-dir outputs/trained
done
```

## Implementation checks

Python syntax compilation passed. On each final model, eight examples from data seed 47821 were used for implementation checks only: all 18 activation sites reproduced the original logits exactly when patched with their own cached activations; changing the final token left every earlier logit unchanged; replacing the final layer's last-position residual output transferred the donor's last-position logits exactly; and key/value uniqueness and target/query metadata invariants passed. Results are recorded in `smoke_checks.json`. These checks establish basic implementation behavior, not causal findings or task generalization.
