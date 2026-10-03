"""CPU-only training; example: .venv/bin/python train.py --seed 0 --steps 3000.

Every training batch is freshly sampled. Repeated evaluation uses one fixed,
independently generated pilot-validation set; it is not confirmatory evidence.
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from model import ModelConfig, TinyTransformer, generate_batch


@torch.no_grad()
def evaluate(model, seed=100000, batches=20, batch_size=256):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss, correct, count = 0.0, 0, 0
    for _ in range(batches):
        tokens, targets, _ = generate_batch(batch_size, generator, model.config)
        logits = model(tokens)[:, -1]
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(-1) == targets).sum().item()
        count += batch_size
    model.train(was_training)
    return {"validation_loss": total_loss / count, "validation_accuracy": correct / count,
            "validation_count": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--validation-seed", type=int, default=100000)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--embedding-std", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"))
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval-every must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    config = ModelConfig(n_heads=args.n_heads, embedding_std=args.embedding_std)
    model = TinyTransformer(config)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"seed_{args.seed}"
    started = time.perf_counter()
    logs = []
    window_loss, window_correct, window_count = 0.0, 0, 0
    for step in range(1, args.steps + 1):
        model.train()
        tokens, targets, _ = generate_batch(args.batch_size, generator, config)
        logits = model(tokens)[:, -1]
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        window_loss += loss.item() * args.batch_size
        window_correct += (logits.detach().argmax(-1) == targets).sum().item()
        window_count += args.batch_size
        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step, "train_loss": window_loss / window_count,
                   "train_accuracy": window_correct / window_count,
                   **evaluate(model, seed=args.validation_seed, batches=args.eval_batches),
                   "elapsed_seconds": time.perf_counter() - started}
            logs.append(row)
            print(json.dumps(row), flush=True)
            with stem.with_suffix(".csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                writer.writeheader()
                writer.writerows(logs)
            window_loss, window_correct, window_count = 0.0, 0, 0
    metadata = {"seed": args.seed, "training_data_seed": args.seed + 1000,
                "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
                "weight_decay": 0.01, "gradient_clip_norm": 1.0, "optimizer": "AdamW",
                "threads": args.threads, "device": "cpu", "config": config.to_dict(),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "validation_seed": args.validation_seed,
                "validation_role": "pilot/discovery only; reused during training",
                "torch_version": str(torch.__version__), "python_version": platform.python_version(),
                "elapsed_seconds": time.perf_counter() - started, "final_metrics": logs[-1]}
    torch.save({"state_dict": model.state_dict(), "config": config.to_dict(),
                "metadata": metadata}, stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(stem.with_suffix(".pt")), **metadata}), flush=True)


if __name__ == "__main__":
    main()
