"""Frozen E2 forecasts from discovery probabilities only; no model execution.

Pure numerical second-order log-odds extrapolation. CLI reads only the fixed
condition list, discovery summary and six discovery NPZ files. It writes one
forecast JSON. It never imports Torch or any experiment/model runner.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

EPS = 1e-6
STRENGTH_FLOOR = 1e-3
SEEDS = tuple(range(6))
KINDS = ("mean", "zero", "resample")
GROUPS = ("pair", "triple_rescue", "cross_layer")


def logodds(values):
    p = np.clip(np.asarray(values, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def probability(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


def forecast_seed(summary, raw, conditions):
    """Return condition means and parameters; arrays must be discovery only."""
    observed = summary["conditions"]
    x0 = logodds(raw["clean/target_probability"])
    fitted, parameters = {}, {}
    for layer in (1, 2):
        for kind in KINDS:
            names = [f"L{layer}_{kind}_h{head}" for head in range(4)]
            changes = np.stack([logodds(raw[n + "/target_probability"]) - x0
                                for n in names], axis=1)
            endpoint = logodds(raw[f"L{layer}_{kind}_all/target_probability"])
            # Positive average lost log odds represents operational head strength.
            # The fixed floor prevents tiny/negative means from making pair weights
            # undefined. No head is selected, dropped or aligned across models.
            strength = np.maximum(-changes.mean(axis=0), STRENGTH_FLOOR)
            weights = strength / strength.sum()
            pair_total = sum(weights[i] * weights[j]
                             for i, j in itertools.combinations(range(4), 2))
            residual = endpoint - x0 - changes.sum(axis=1)
            fitted[layer, kind] = (changes, weights, pair_total, residual)
            parameters[f"L{layer}_{kind}"] = {
                "head_strength": strength.tolist(),
                "head_weight": weights.tolist(),
                "mean_single_logodds_change": changes.mean(axis=0).tolist(),
                "mean_all_head_interaction_residual": float(residual.mean()),
            }
    forecasts = {}
    for condition in conditions:
        name = condition["id"]
        if name in observed:
            forecasts[name] = float(observed[name]["target_probability"]["mean"])
            continue
        if condition["group"] not in GROUPS:
            raise ValueError(f"Unrecognized unseen condition: {name}")
        latent = x0.copy()
        for op in condition["ops"]:
            if op.get("component", "z") != "z":
                raise ValueError("No unseen MLP rule was preregistered")
            layer, kind = op["layer"] + 1, op.get("corruption", op["kind"])
            selected = ([h for h in range(4) if h not in op["heads"]]
                        if op["kind"] == "rescue" else list(op["heads"]))
            changes, weights, pair_total, residual = fitted[layer, kind]
            pair_weight = sum(weights[i] * weights[j]
                              for i, j in itertools.combinations(selected, 2)) / pair_total
            latent += changes[:, selected].sum(axis=1) + pair_weight * residual
        forecasts[name] = float(probability(latent).mean())
    if len(forecasts) != len(conditions) or not all(np.isfinite(list(forecasts.values()))):
        raise ValueError("Missing, duplicate or nonfinite forecast")
    if not all(0 <= p <= 1 for p in forecasts.values()):
        raise ValueError("Forecast outside probability support")
    return forecasts, parameters


def generate(conditions_path, discovery_dir):
    conditions = json.loads(conditions_path.read_text())["confirmatory"]
    summary_path = discovery_dir / "summary.json"
    summaries = json.loads(summary_path.read_text())
    probabilities, parameters = {}, {}
    sources = [conditions_path, summary_path]
    for seed in SEEDS:
        path = discovery_dir / f"seed_{seed}.npz"
        sources.append(path)
        with np.load(path, allow_pickle=False) as raw:
            probabilities[str(seed)], parameters[str(seed)] = forecast_seed(
                summaries[str(seed)], raw, conditions)
    return {
        "model": "discovery-only endpoint-constrained second-order log-odds",
        "endpoint": "condition mean original-target probability",
        "expected_probability": probabilities,
        "parameters": parameters,
        "fixed_constants": {"probability_clip": EPS, "head_strength_floor": STRENGTH_FLOOR},
        "practical_bounds": {"absolute_error": 0.10,
                             "interpretation": "[max(0,prediction-0.10), min(1,prediction+0.10)]; not a coverage claim"},
        "success_gates": {
            "each_seed_balanced_novel_rmse_max": 0.10,
            "each_seed_novel_fraction_within_0_10_min": 0.80,
            "each_seed_novel_count_within_0_10_min": 74,
            "baseline_advantage_each_seed_bootstrap_ci95_upper_strictly_below": 0.0,
            "global_claim_requires_all_six_seeds": True,
            "validity_clean_accuracy_min": 0.95,
            "validity_runtime_noop_max_probability_error": 1e-6,
        },
        "primary_groups_equal_weight": list(GROUPS),
        "bootstrap": {"replicates": 2000, "seed": 6200001,
                      "shared_case_weights_across_seeds": True,
                      "baseline_comparison": "AI MSE minus minimum of seven baseline MSEs in each resample"},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, default=Path("experiment2/conditions.json"))
    parser.add_argument("--discovery", type=Path, default=Path("outputs/experiment2/discovery"))
    parser.add_argument("--output", type=Path, default=Path("experiment2/predictions.json"))
    args = parser.parse_args()
    result = generate(args.conditions, args.discovery)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"models": len(result["expected_probability"]),
                      "forecasts": sum(map(len, result["expected_probability"].values()))}))


if __name__ == "__main__":
    main()
