"""Frozen E3 scoring from saved arrays only; no model imports or forwards."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("outputs/experiment3")
FORECAST_METRICS = ("original_probability", "slot_probability", "original_accuracy", "slot_accuracy")
ATTENTION_METRICS = ("l1_true_key_attention", "l1_slot_key_attention",
                     "l2_original_value_attention", "l2_slot_value_attention")


def stratified_weights(query_pair, replicates, seed):
    """Bootstrap dictionaries within query strata, retaining equal stratum mass.

    One weight matrix is reused across every condition and model seed. This
    preserves pairing; a row's outcomes are never resampled independently.
    """
    query_pair = np.asarray(query_pair)
    if query_pair.ndim != 1 or not np.isin(query_pair, range(4)).all():
        raise ValueError("Expected query positions 0..3")
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(query_pair)), dtype=np.float64)
    for query in range(4):
        indices = np.flatnonzero(query_pair == query)
        if not len(indices):
            raise ValueError("Every query stratum must be present")
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                             size=replicates) / (4 * len(indices))
    return weights


def ci(draws):
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def wilson(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if not n or not np.isin(values, [0, 1]).all():
        raise ValueError("Wilson interval requires nonempty binary observations")
    p, z = float(values.mean()), 1.959963984540054
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return {"mean": p, "n": n, "ci95_wilson": [float(center - radius), float(center + radius)]}


def bootstrap_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "ci95_bootstrap": ci(weights @ values)}


def forecast_stats(rows):
    errors = np.array([row["error"] for row in rows], dtype=np.float64)
    if not len(errors):
        return {"count": 0}
    return {"count": len(rows), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "within_tolerance": sum(row["within_tolerance"] for row in rows),
            "fraction_within_tolerance": float(np.mean([row["within_tolerance"] for row in rows])),
            "max_absolute_error": float(np.abs(errors).max())}


def score_seed(raw, arrays, conditions, forecasts, protocol, weights):
    """Score one seed with supplied shared bootstrap weights; supports fixtures."""
    targets = np.asarray(arrays["targets"], dtype=np.int64)
    query_pair = np.asarray(arrays["query_pair"], dtype=np.int64)
    values = np.asarray(arrays["values"], dtype=np.int64)
    n = len(targets)
    names = [condition["id"] for condition in conditions]
    if len(set(names)) != len(names) or weights.shape[1] != n:
        raise ValueError("Invalid condition ids or bootstrap weights")
    by_id = {condition["id"]: condition for condition in conditions}
    probability = np.stack([np.asarray(raw[name + "/" + metric], dtype=np.float64)
                            for name in names for metric in FORECAST_METRICS[:2]], axis=1)
    if probability.shape != (n, len(names) * 2) or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Invalid probability arrays")
    draws = weights @ probability
    intervals = np.quantile(draws, [.025, .975], axis=0)
    query_intervals = {}
    for query in range(4):
        mask = query_pair == query
        query_intervals[query] = np.quantile((weights[:, mask] * 4) @ probability[mask], [.025, .975], axis=0)
    output = {"n": n, "query_counts": np.bincount(query_pair, minlength=4).tolist(),
              "conditions": {}, "primary_conditions": {}, "baseline_fidelity": {}}
    forecast_rows = []
    for index, name in enumerate(names):
        condition = by_id[name]
        predicted = np.asarray(raw[name + "/argmax_class"], dtype=np.int64)
        if predicted.shape != (n,) or not np.isin(predicted, range(16)).all():
            raise ValueError("Invalid argmax classes")
        slot_pairs = np.asarray(condition["slot_value_by_query"])[query_pair]
        slot_targets = values[np.arange(n), slot_pairs]
        class_probability = np.asarray(raw[name + "/class_probability"], dtype=np.float64)
        if (class_probability.shape != (n, 16) or not np.isfinite(class_probability).all()
                or not ((class_probability >= 0) & (class_probability <= 1)).all()
                or not np.allclose(class_probability.sum(1), 1, rtol=1e-6, atol=1e-7)
                or not np.array_equal(class_probability.argmax(1), predicted)):
            raise ValueError(f"{name}: invalid full class probabilities")
        for metric, classes in (("original_probability", targets), ("slot_probability", slot_targets)):
            if not np.allclose(class_probability[np.arange(n), classes], raw[name + "/" + metric], rtol=0, atol=1e-8):
                raise ValueError(f"{name}: full probabilities disagree with {metric}")
        accuracy = {"original_accuracy": (predicted == targets).astype(np.int64),
                    "slot_accuracy": (predicted == slot_targets).astype(np.int64)}
        for metric, expected in accuracy.items():
            if not np.array_equal(expected, raw[name + "/" + metric]):
                raise ValueError(f"{name}/{metric}: saved accuracy disagrees with argmax")
        row = {"family": condition["family"], "primary_group": condition["primary_group"],
               "slot_mapping_defined": condition["slot_mapping_defined"], "metrics": {},
               "per_query": {}, "attention": {}}
        for offset, metric in enumerate(FORECAST_METRICS[:2]):
            column = 2 * index + offset
            row["metrics"][metric] = {"mean": float(probability[:, column].mean()),
                                      "ci95_bootstrap": intervals[:, column].tolist()}
        for metric, array in accuracy.items():
            row["metrics"][metric] = wilson(array)
        for query in range(4):
            mask = query_pair == query
            query_row = {"n": int(mask.sum())}
            for offset, metric in enumerate(FORECAST_METRICS[:2]):
                column = 2 * index + offset
                query_row[metric] = {"mean": float(probability[mask, column].mean()),
                                     "ci95_bootstrap": query_intervals[query][:, column].tolist()}
            for metric, array in accuracy.items():
                query_row[metric] = wilson(array[mask])
            row["per_query"][str(query)] = query_row
        for metric in ATTENTION_METRICS:
            array = np.asarray(raw[name + "/" + metric], dtype=np.float64)
            if array.shape != (n, 4) or not np.isfinite(array).all() or not ((array >= 0) & (array <= 1)).all():
                raise ValueError(f"Invalid attention diagnostic {name}/{metric}")
            row["attention"][metric] = {"heads": array.mean(0).tolist(),
                                        "equal_head_mean": float(array.mean()),
                                        "per_query_heads": {str(q): array[query_pair == q].mean(0).tolist() for q in range(4)}}
        fidelity = {"semantic_original": accuracy["original_accuracy"].astype(float),
                    "uniform_displayed_four": .25 * np.any(predicted[:, None] == values, axis=1),
                    "uniform_sixteen": np.full(n, 1 / 16),
                    "slot_rule": accuracy["slot_accuracy"].astype(float)}
        output["baseline_fidelity"][name] = {method: bootstrap_mean(array, weights) for method, array in fidelity.items()}
        coincidence = float(np.mean(slot_targets == targets))
        assigned = {"semantic_original": (1.0, coincidence), "uniform_displayed_four": (.25, .25),
                    "uniform_sixteen": (1 / 16, 1 / 16), "slot_rule": (coincidence, 1.0)}
        for method, (original_assigned, slot_assigned) in assigned.items():
            output["baseline_fidelity"][name][method].update({
                "assigned_original_probability": original_assigned, "assigned_slot_probability": slot_assigned,
                "original_probability_error": row["metrics"]["original_probability"]["mean"] - original_assigned,
                "slot_probability_error": row["metrics"]["slot_probability"]["mean"] - slot_assigned})
        output["conditions"][name] = row
        for metric in FORECAST_METRICS:
            expected = float(forecasts[metric][name])
            observed = row["metrics"][metric]["mean"]
            error = observed - expected
            forecast_rows.append({"condition": name, "family": condition["family"],
                                  "primary_group": condition["primary_group"], "metric": metric,
                                  "forecast": expected, "observed": observed, "error": error,
                                  "absolute_error": abs(error),
                                  "within_tolerance": abs(error) <= protocol["numeric_forecast_tolerance"]})
    value_names = [c["id"] for c in conditions if c["primary_group"] == "value_derangement"]
    coherent_names = [c["id"] for c in conditions if c["primary_group"] == "coherent_control"]
    if len(value_names) != 9 or len(coherent_names) != 9:
        raise ValueError("Primary analysis requires nine value and nine coherent derangements")
    thresholds = protocol["gates"]
    for name in value_names:
        metric = output["conditions"][name]["metrics"]["slot_accuracy"]
        output["primary_conditions"][name] = {"endpoint": "slot_accuracy", **metric,
            "threshold": thresholds["each_derangement_slot_accuracy_min"],
            "pass": metric["mean"] >= thresholds["each_derangement_slot_accuracy_min"]}
    for name in coherent_names:
        metric = output["conditions"][name]["metrics"]["original_accuracy"]
        output["primary_conditions"][name] = {"endpoint": "original_accuracy", **metric,
            "threshold": thresholds["each_matched_coherent_original_accuracy_min"],
            "pass": metric["mean"] >= thresholds["each_matched_coherent_original_accuracy_min"]}
    margins = np.stack([np.asarray(raw[name + "/slot_probability"]) - np.asarray(raw[name + "/original_probability"])
                        for name in value_names], axis=1)
    row_margin = margins.mean(1)
    output["primary_probability_margin"] = bootstrap_mean(row_margin, weights)
    output["primary_probability_margin"]["per_query"] = {
        str(q): bootstrap_mean(row_margin[query_pair == q], weights[:, query_pair == q] * 4) for q in range(4)}
    noop_error = float(np.max(np.abs(np.asarray(raw["original_native/original_probability"]) - np.asarray(raw["original_noop/original_probability"]))))
    output["noop_max_probability_error"] = noop_error
    output["validity_gates"] = {
        "native_accuracy": output["conditions"]["original_native"]["metrics"]["original_accuracy"]["mean"] >= thresholds["original_native_accuracy_min"],
        "noop_identity": noop_error <= thresholds["original_noop_max_probability_error"]}
    output["mechanism_gates"] = {
        "every_value_derangement": all(output["primary_conditions"][name]["pass"] for name in value_names),
        "every_coherent_control": all(output["primary_conditions"][name]["pass"] for name in coherent_names),
        "probability_margin_ci95_lower": output["primary_probability_margin"]["ci95_bootstrap"][0] > thresholds["each_seed_mean_derangement_slot_minus_original_probability_ci95_lower_gt"]}
    output["forecast_statistics"] = {"all": forecast_stats(forecast_rows),
        "primary_18": forecast_stats([row for row in forecast_rows if row["primary_group"] is not None]),
        "primary_value_9": forecast_stats([row for row in forecast_rows if row["primary_group"] == "value_derangement"])}
    # Secondary inverse-vs-forward check: exclude queries on which the rules coincide.
    secondary = {}
    inverse_sum, forward_sum, eligible_count = np.zeros(n), np.zeros(n), np.zeros(n)
    inverse_probability_sum, forward_probability_sum = np.zeros(n), np.zeros(n)
    for condition in conditions:
        if condition["family"] != "value_only":
            continue
        pi = np.asarray(condition["permutation"])
        inverse = np.asarray(condition["slot_value_by_query"])
        if np.array_equal(pi, inverse):
            continue
        name = condition["id"]
        eligible = pi[query_pair] != inverse[query_pair]
        predicted = np.asarray(raw[name + "/argmax_class"])
        forward_target = values[np.arange(n), pi[query_pair]]
        inverse_target = values[np.arange(n), inverse[query_pair]]
        inv = (predicted == inverse_target).astype(float)
        fwd = (predicted == forward_target).astype(float)
        class_probability = np.asarray(raw[name + "/class_probability"])
        inverse_probability = class_probability[np.arange(n), inverse_target]
        forward_probability = class_probability[np.arange(n), forward_target]
        inverse_sum += inv * eligible
        forward_sum += fwd * eligible
        inverse_probability_sum += inverse_probability * eligible
        forward_probability_sum += forward_probability * eligible
        eligible_count += eligible
        secondary[name] = {"eligible_cases": int(eligible.sum()),
                            "inverse_accuracy": wilson(inv[eligible]), "forward_accuracy": wilson(fwd[eligible]),
                            "inverse_probability": float(inverse_probability[eligible].mean()),
                            "forward_probability": float(forward_probability[eligible].mean()),
                            "per_query": {str(q): {"n": int(np.sum(eligible & (query_pair == q))),
                                "inverse_accuracy": float(inv[eligible & (query_pair == q)].mean()),
                                "forward_accuracy": float(fwd[eligible & (query_pair == q)].mean())}
                                for q in range(4) if np.any(eligible & (query_pair == q))}}
    if len(secondary) != 14 or np.any(eligible_count == 0):
        raise ValueError("Expected 14 non-involutions with eligible comparisons")
    inverse_row, forward_row = inverse_sum / eligible_count, forward_sum / eligible_count
    output["secondary_forward_inverse"] = {"conditions": secondary,
        "definition": "Eligible non-involution conditions averaged within dictionary row, then query-stratified paired bootstrap; descriptive only",
        "eligible_conditions_per_row": sorted(set(eligible_count.astype(int).tolist())),
        "inverse_fidelity": bootstrap_mean(inverse_row, weights),
        "forward_fidelity": bootstrap_mean(forward_row, weights),
        "inverse_minus_forward": bootstrap_mean(inverse_row - forward_row, weights),
        "inverse_probability": bootstrap_mean(inverse_probability_sum / eligible_count, weights),
        "forward_probability": bootstrap_mean(forward_probability_sum / eligible_count, weights),
        "inverse_minus_forward_probability": bootstrap_mean((inverse_probability_sum - forward_probability_sum) / eligible_count, weights)}
    wrong_target_rows, coherent_wrong_rows, other_wrong_rows = [], [], []
    for name in value_names:
        condition = by_id[name]
        wrong_pair = np.asarray(condition["slot_value_by_query"])[query_pair]
        wrong_class = values[np.arange(n), wrong_pair]
        coherent_name = "coherent_" + name.split("_", 1)[1]
        full_probability = np.asarray(raw[name + "/class_probability"])
        wrong_target_rows.append(full_probability[np.arange(n), wrong_class])
        coherent_wrong_rows.append(np.asarray(raw[coherent_name + "/class_probability"])[np.arange(n), wrong_class])
        displayed = full_probability[np.arange(n)[:, None], values]
        others = (np.arange(4)[None, :] != query_pair[:, None]) & (np.arange(4)[None, :] != wrong_pair[:, None])
        other_wrong_rows.append(np.sum(displayed * others, axis=1) / 2)
    value_wrong, coherent_wrong = np.mean(wrong_target_rows, axis=0), np.mean(coherent_wrong_rows, axis=0)
    output["matched_coherent_wrong_target"] = {
        "definition": "Same inverse-pi wrong target in each of nine paired value/coherent derangements; average within dictionary before bootstrap; descriptive only",
        "value_condition_wrong_target_probability": bootstrap_mean(value_wrong, weights),
        "coherent_condition_wrong_target_probability": bootstrap_mean(coherent_wrong, weights),
        "other_two_wrong_displayed_targets_mean_probability": bootstrap_mean(np.mean(other_wrong_rows, axis=0), weights),
        "paired_difference": bootstrap_mean(value_wrong - coherent_wrong, weights)}
    output["secondary_probability_contrasts"] = {
        "canonical_minus_native_grouped_original_probability": bootstrap_mean(
            np.asarray(raw["grouped_canonical/original_probability"]) - np.asarray(raw["grouped_native/original_probability"]), weights),
        "coherent_minus_value_original_probability": {
            c["id"]: bootstrap_mean(np.asarray(raw["coherent_" + c["id"].split("_", 1)[1] + "/original_probability"])
                                      - np.asarray(raw[c["id"] + "/original_probability"]), weights)
            for c in conditions if c["family"] == "value_only"}}
    return output, forecast_rows


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = json.loads(Path("experiment3/protocol.json").read_text())
    conditions = json.loads(Path("experiment3/conditions.json").read_text())
    predictions = json.loads(Path("experiment3/predictions.json").read_text())
    if not (ROOT / "confirmatory/summary.json").is_file():
        raise SystemExit("Completed confirmation summary required before scoring")
    with np.load(ROOT / "inputs.npz") as input_file:
        arrays = {name: input_file[name] for name in input_file.files}
    n = protocol["n"]
    if len(arrays["targets"]) != n or not np.array_equal(np.bincount(arrays["query_pair"], minlength=4), np.full(4, n // 4)):
        raise ValueError("Expected the registered balanced input set")
    bootstrap = protocol["bootstrap"]
    weights = stratified_weights(arrays["query_pair"], bootstrap["replicates"], bootstrap["seed"])
    scores, forecast_rows, baseline_rows, attention_rows = {}, [], [], []
    for seed in protocol["model_seeds"]:
        s = str(seed)
        forecasts = {metric: predictions["expected_" + metric][s] for metric in FORECAST_METRICS}
        with np.load(ROOT / f"confirmatory/seed_{seed}.npz") as raw:
            row, forecast = score_seed(raw, arrays, conditions, forecasts, protocol, weights)
        scores[s] = row
        forecast_rows.extend({"seed": seed, **item} for item in forecast)
        for condition, methods in row["baseline_fidelity"].items():
            for method, result in methods.items():
                baseline_rows.append({"seed": seed, "condition": condition, "baseline": method,
                                      "answer_fidelity": result["mean"], "ci95_low": result["ci95_bootstrap"][0],
                                      "ci95_high": result["ci95_bootstrap"][1],
                                      **{key: result[key] for key in ("assigned_original_probability", "assigned_slot_probability",
                                          "original_probability_error", "slot_probability_error")}})
        for condition, result in row["conditions"].items():
            for metric, diagnostic in result["attention"].items():
                for head, value in enumerate(diagnostic["heads"]):
                    attention_rows.append({"seed": seed, "condition": condition, "diagnostic": metric,
                                           "head": head, "mean_attention": value})
        print(json.dumps({"seed": seed, "validity": row["validity_gates"],
                          "mechanism": row["mechanism_gates"], "margin": row["primary_probability_margin"]}), flush=True)
    decisions = {"all_six_valid": all(all(row["validity_gates"].values()) for row in scores.values()),
                 "all_six_mechanism_gates": all(all(row["mechanism_gates"].values()) for row in scores.values())}
    decisions["full_claim_pass"] = all(decisions.values())
    (ROOT / "confirmatory/scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (ROOT / "confirmatory/global_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    numeric = {"all": forecast_stats(forecast_rows),
               "primary_18": forecast_stats([row for row in forecast_rows if row["primary_group"] is not None]),
               "primary_value_9": forecast_stats([row for row in forecast_rows if row["primary_group"] == "value_derangement"])}
    (ROOT / "confirmatory/forecast_statistics.json").write_text(json.dumps(numeric, indent=2) + "\n")
    write_csv(ROOT / "forecasts_vs_results.csv", forecast_rows)
    misses = [row for row in forecast_rows if not row["within_tolerance"]]
    write_csv(ROOT / "forecast_misses.csv", misses, fields=list(forecast_rows[0]))
    write_csv(ROOT / "baseline_fidelity.csv", baseline_rows)
    write_csv(ROOT / "attention_diagnostics.csv", attention_rows)


if __name__ == "__main__":
    main()
