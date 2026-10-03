"""Standalone E4 postprocessing. Reads saved outcomes; never executes models."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("outputs/experiment4")
STATE_METRICS = ("l1_value_max_error_native", "l1_query_max_error_native", "l1_key_max_error_native",
                 "l1_key_max_error_grouped", "l2_input_value_max_error_native", "l2_input_query_max_error_native",
                 "l2_input_key_max_error_native", "l2_input_key_max_error_grouped")
ATTENTION_METRICS = ("l1_true_key_attention", "l2_correct_value_attention")


def stratified_weights(query_pair, replicates, seed):
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


def lower_exceeds(bounds, threshold):
    return bounds[0] > threshold


def mean_ci(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "ci95_bootstrap": ci(weights @ values)}


def wilson(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if not n or not np.isin(values, [0, 1]).all():
        raise ValueError("Wilson interval requires binary observations")
    p, z = float(values.mean()), 1.959963984540054
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return {"mean": p, "n": n, "ci95_wilson": [float(center - radius), float(center + radius)]}


def forecast_stats(rows):
    if not rows:
        return {"count": 0}
    errors = np.asarray([row["error"] for row in rows])
    return {"count": len(rows), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "within_tolerance": sum(row["within_tolerance"] for row in rows),
            "fraction_within_tolerance": float(np.mean([row["within_tolerance"] for row in rows])),
            "max_absolute_error": float(np.abs(errors).max())}


def score_seed(raw, arrays, conditions, forecasts, protocol, weights):
    targets, query = np.asarray(arrays["targets"]), np.asarray(arrays["query_pair"])
    n = len(targets)
    names = [condition["id"] for condition in conditions]
    index = {name: i for i, name in enumerate(names)}
    if len(names) != 15 or len(index) != 15 or weights.shape[1] != n:
        raise ValueError("Expected fifteen conditions and matching shared weights")
    probabilities, accuracies, class_probs = [], {}, {}
    for name in names:
        full = np.asarray(raw[name + "/class_probability"], dtype=np.float64)
        predicted = np.asarray(raw[name + "/argmax_class"])
        probability = np.asarray(raw[name + "/original_probability"], dtype=np.float64)
        accuracy = np.asarray(raw[name + "/accuracy"])
        if (full.shape != (n, 16) or not np.isfinite(full).all() or not ((full >= 0) & (full <= 1)).all()
                or not np.allclose(full.sum(1), 1, rtol=1e-6, atol=1e-7)
                or predicted.shape != (n,) or not np.array_equal(predicted, full.argmax(1))
                or probability.shape != (n,) or not np.isfinite(probability).all()
                or not np.allclose(probability, full[np.arange(n), targets], rtol=0, atol=1e-8)
                or accuracy.shape != (n,) or not np.array_equal(accuracy, (predicted == targets).astype(int))):
            raise ValueError(f"{name}: inconsistent output arrays")
        probabilities.append(probability)
        accuracies[name] = accuracy
        class_probs[name] = full
    probabilities = np.stack(probabilities, axis=1)
    bootstrap = weights @ probabilities
    bounds = np.quantile(bootstrap, [.025, .975], axis=0)
    query_bounds = {q: np.quantile((weights[:, query == q] * 4) @ probabilities[query == q], [.025, .975], axis=0)
                    for q in range(4)}
    output = {"n": n, "query_counts": np.bincount(query, minlength=4).tolist(), "conditions": {}}
    forecast_rows = []
    for condition in conditions:
        name, column = condition["id"], index[condition["id"]]
        result = {"family": condition["family"], "oracle": condition["oracle"],
                  "original_probability": {"mean": float(probabilities[:, column].mean()), "ci95_bootstrap": bounds[:, column].tolist()},
                  "accuracy": wilson(accuracies[name]), "per_query": {}, "state_errors": {}, "attention": {}}
        for q in range(4):
            selected = query == q
            result["per_query"][str(q)] = {
                "original_probability": {"mean": float(probabilities[selected, column].mean()),
                                          "ci95_bootstrap": query_bounds[q][:, column].tolist()},
                "accuracy": wilson(accuracies[name][selected])}
        for metric in STATE_METRICS:
            values = np.asarray(raw[name + "/" + metric], dtype=np.float64)
            if values.shape != (n,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name}: invalid residual diagnostic {metric}")
            result["state_errors"][metric] = {"mean": float(values.mean()), "max": float(values.max())}
        for metric in ATTENTION_METRICS:
            values = np.asarray(raw[name + "/" + metric], dtype=np.float64)
            if values.shape != (n, 4) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError(f"{name}: invalid attention diagnostic {metric}")
            result["attention"][metric] = {"heads": values.mean(0).tolist(), "equal_head_mean": float(values.mean())}
        output["conditions"][name] = result
        for metric in ("original_probability", "accuracy"):
            observed = result[metric]["mean"]
            predicted = float(forecasts[metric][name])
            error = observed - predicted
            forecast_rows.append({"condition": name, "family": condition["family"], "oracle": condition["oracle"],
                                  "metric": metric, "forecast": predicted, "observed": observed,
                                  "error": error, "absolute_error": abs(error),
                                  "within_tolerance": abs(error) <= protocol["numeric_forecast_tolerance"]})
    guard_name = protocol["correct_guard"]
    wrong_names = [condition["id"] for condition in conditions
                   if condition["family"] == "matched_guard" and condition["id"] != guard_name]
    if len(wrong_names) != 8:
        raise ValueError("Exactly eight wrong row-degree-matched masks required")
    guard = probabilities[:, index[guard_name]]
    grouped = probabilities[:, index["grouped_canonical"]]
    wrong = probabilities[:, [index[name] for name in wrong_names]]
    mean_wrong = wrong.mean(1)
    best_column = int(wrong.mean(0).argmax())
    best_name = wrong_names[best_column]
    guard_boot = bootstrap[:, index[guard_name]]
    wrong_boot = bootstrap[:, [index[name] for name in wrong_names]]
    comparisons = {"guard_minus_grouped": mean_ci(guard - grouped, weights),
                   "guard_minus_mean_eight": mean_ci(guard - mean_wrong, weights),
                   "mean_eight_wrong_probability": mean_ci(mean_wrong, weights),
                   "guard_minus_best_wrong": {"best_point_condition": best_name,
                       "best_point_probability": float(wrong[:, best_column].mean()),
                       "mean": float(guard.mean() - wrong[:, best_column].mean()),
                       "ci95_bootstrap": ci(guard_boot - wrong_boot.max(1)),
                       "definition": "Best wrong-mask mean is reselected in each shared paired bootstrap; descriptive only"},
                   "guard_minus_each_wrong": {name: mean_ci(guard - wrong[:, i], weights) for i, name in enumerate(wrong_names)},
                   "per_query": {str(q): {
                       "guard_minus_grouped": mean_ci((guard - grouped)[query == q], weights[:, query == q] * 4),
                       "guard_minus_mean_eight": mean_ci((guard - mean_wrong)[query == q], weights[:, query == q] * 4)} for q in range(4)}}
    output["comparisons"] = comparisons
    keys = probabilities[:, index["restore_keys"]]
    full = probabilities[:, index["guard_restore_keys"]]
    output["factorial_descriptive"] = {
        "keys_effect_without_guard": mean_ci(keys - grouped, weights),
        "keys_effect_with_guard": mean_ci(full - guard, weights),
        "guard_effect_without_keys": mean_ci(guard - grouped, weights),
        "guard_effect_with_keys": mean_ci(full - keys, weights),
        "interaction_full_minus_guard_minus_keys_plus_grouped": mean_ci(full - guard - keys + grouped, weights),
        "interpretation": "Paired probability-scale two-factor contrast; no additional gate"}
    errors = {
        "grouped_noop_probability": float(np.max(np.abs(probabilities[:, index["grouped_noop"]] - grouped))),
        "correct_guard_l1_value_native": float(np.max(raw[guard_name + "/l1_value_max_error_native"])),
        "correct_guard_l1_query_native": float(np.max(raw[guard_name + "/l1_query_max_error_native"])),
        "correct_guard_l1_key_grouped": float(np.max(raw[guard_name + "/l1_key_max_error_grouped"])),
        "guard_vs_value_oracle_full_probability": float(np.max(np.abs(class_probs[guard_name] - class_probs["restore_values"]))),
        "full_oracle_vs_native_full_probability": float(np.max(np.abs(class_probs["guard_restore_keys"] - class_probs["native"])))}
    output["manipulation_errors"] = errors
    thresholds = protocol["gates"]
    output["validity_gates"] = {
        "native_accuracy": output["conditions"]["native"]["accuracy"]["mean"] >= thresholds["native_accuracy_min"],
        "finite_complete_outputs": True,
        "grouped_noop_identity": errors["grouped_noop_probability"] <= thresholds["grouped_noop_max_probability_error"],
        **{name: error <= thresholds["identity_tolerance"] for name, error in errors.items() if name != "grouped_noop_probability"}}
    output["query_recovery_gates"] = {str(q): output["conditions"][guard_name]["per_query"][str(q)]["accuracy"]["mean"]
                                     >= thresholds["correct_guard_each_query_accuracy_min"] for q in range(4)}
    output["recovery_gates"] = {"each_query_accuracy": all(output["query_recovery_gates"].values()),
        "guard_minus_grouped_ci95_lower": lower_exceeds(comparisons["guard_minus_grouped"]["ci95_bootstrap"],
            thresholds["guard_minus_grouped_probability_ci95_lower_gt"])}
    output["specificity_gates"] = {"guard_minus_mean_eight_ci95_lower": lower_exceeds(comparisons["guard_minus_mean_eight"]["ci95_bootstrap"],
            thresholds["guard_minus_mean_eight_controls_probability_ci95_lower_gt"])}
    output["forecast_statistics"] = {"all_15": forecast_stats(forecast_rows),
        "matched_guards_9": forecast_stats([row for row in forecast_rows if row["family"] == "matched_guard"]),
        "oracles_3": forecast_stats([row for row in forecast_rows if row["oracle"]]),
        "references_3": forecast_stats([row for row in forecast_rows if row["family"] == "reference"])}
    return output, forecast_rows


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = json.loads(Path("experiment4/protocol.json").read_text())
    conditions = json.loads(Path("experiment4/conditions.json").read_text())
    predictions = json.loads(Path("experiment4/predictions.json").read_text())
    if not (ROOT / "confirmatory/summary.json").is_file():
        raise SystemExit("Completed E4 confirmation summary required")
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    n = protocol["n"]
    if len(arrays["targets"]) != n or not np.array_equal(np.bincount(arrays["query_pair"], minlength=4), np.full(4, n // 4)):
        raise ValueError("Expected registered query-balanced input set")
    bootstrap = protocol["bootstrap"]
    weights = stratified_weights(arrays["query_pair"], bootstrap["replicates"], bootstrap["seed"])
    scores, forecasts, condition_rows, query_rows, attention_rows = {}, [], [], [], []
    for seed in protocol["model_seeds"]:
        s = str(seed)
        expected = {"original_probability": predictions["expected_probability"][s], "accuracy": predictions["expected_accuracy"][s]}
        with np.load(ROOT / f"confirmatory/seed_{seed}.npz", allow_pickle=False) as raw:
            score, rows = score_seed(raw, arrays, conditions, expected, protocol, weights)
        scores[s] = score
        forecasts.extend({"seed": seed, **row} for row in rows)
        for name, result in score["conditions"].items():
            condition_rows.append({"seed": seed, "condition": name, "family": result["family"], "oracle": result["oracle"],
                                  "probability": result["original_probability"]["mean"], "probability_ci95_low": result["original_probability"]["ci95_bootstrap"][0],
                                  "probability_ci95_high": result["original_probability"]["ci95_bootstrap"][1],
                                  "accuracy": result["accuracy"]["mean"], "wilson_low": result["accuracy"]["ci95_wilson"][0],
                                  "wilson_high": result["accuracy"]["ci95_wilson"][1]})
            for q, row in result["per_query"].items():
                query_rows.append({"seed": seed, "condition": name, "query_pair": q,
                                   "probability": row["original_probability"]["mean"], "probability_ci95_low": row["original_probability"]["ci95_bootstrap"][0],
                                   "probability_ci95_high": row["original_probability"]["ci95_bootstrap"][1],
                                   "accuracy": row["accuracy"]["mean"], "wilson_low": row["accuracy"]["ci95_wilson"][0],
                                   "wilson_high": row["accuracy"]["ci95_wilson"][1]})
            for metric, diagnostic in result["attention"].items():
                attention_rows.extend({"seed": seed, "condition": name, "diagnostic": metric, "head": head, "mean_attention": value}
                                      for head, value in enumerate(diagnostic["heads"]))
        print(json.dumps({"seed": seed, "validity": score["validity_gates"], "recovery": score["recovery_gates"],
                          "specificity": score["specificity_gates"]}), flush=True)
    decisions = {"all_six_valid": all(all(row["validity_gates"].values()) for row in scores.values()),
                 "all_six_recovery": all(all(row["recovery_gates"].values()) for row in scores.values()),
                 "all_six_specificity": all(all(row["specificity_gates"].values()) for row in scores.values())}
    decisions["full_claim_pass"] = all(decisions.values())
    numeric = {"all_15": forecast_stats(forecasts),
               "matched_guards_9": forecast_stats([row for row in forecasts if row["family"] == "matched_guard"]),
               "oracles_3": forecast_stats([row for row in forecasts if row["oracle"]]),
               "references_3": forecast_stats([row for row in forecasts if row["family"] == "reference"])}
    for name, content in (("scores", scores), ("global_decisions", decisions), ("forecast_statistics", numeric)):
        (ROOT / f"confirmatory/{name}.json").write_text(json.dumps(content, indent=2) + "\n")
    write_csv(ROOT / "forecasts_vs_results.csv", forecasts)
    write_csv(ROOT / "forecast_misses.csv", [row for row in forecasts if not row["within_tolerance"]], fields=list(forecasts[0]))
    write_csv(ROOT / "condition_results.csv", condition_rows)
    write_csv(ROOT / "per_query_results.csv", query_rows)
    write_csv(ROOT / "attention_diagnostics.csv", attention_rows)


if __name__ == "__main__":
    main()
