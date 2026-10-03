"""Streaming E5 postprocessing only: no model imports or inference."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("outputs/experiment5")
STATE_METRICS = ("l1_value_max_error_native", "l1_query_max_error_native", "l1_key_max_error_native",
                 "l1_key_max_error_baseline", "l1_value_max_error_own_key_self",
                 "l2_input_value_max_error_native", "l2_input_query_max_error_native",
                 "l2_input_key_max_error_native", "class_probability_max_error_native")
METRICS = ("original_probability", "accuracy", "argmax_class", "class_probability") + STATE_METRICS


def stratified_weights(query, replicates, seed):
    query = np.asarray(query)
    if query.ndim != 1 or not np.isin(query, range(4)).all():
        raise ValueError("Invalid query strata")
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(query)), dtype=np.float64)
    for q in range(4):
        indices = np.flatnonzero(query == q)
        if not len(indices):
            raise ValueError("All four query strata required")
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1/len(indices)), size=replicates)/(4*len(indices))
    return weights


def mean_ci(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "ci95": np.quantile(weights @ values, [.025, .975]).tolist()}


def wilson(values):
    values = np.asarray(values)
    if not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError("Wilson requires nonempty binary observations")
    n, p, z = len(values), float(values.mean()), 1.959963984540054
    scale = 1+z*z/n
    center = (p+z*z/(2*n))/scale
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return [float(center-radius), float(center+radius)]


def iter_saved_panel(seed, panel, root=ROOT):
    """Yield (layout_id, condition->metric->array) from one shard at a time."""
    paths = sorted((root / f"confirmatory/seed_{seed}" / panel).glob("shard_*.npz"))
    if not paths:
        raise ValueError(f"Missing saved panel {seed}/{panel}")
    ledger_path = root / f"confirmatory/seed_{seed}/index.json"
    ledger = json.loads(ledger_path.read_text())
    entries = ledger["panels"][panel]["shards"]
    if ledger["seed"] != seed or {str(p) for p in paths} != {entry["path"] for entry in entries} or len(entries) != len(paths):
        raise ValueError("Shard index/path coverage mismatch")
    indexed = {entry["path"]: entry for entry in entries}
    seen = set()
    for path in paths:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != indexed[str(path)]["sha256"]:
                raise ValueError("Shard hash differs from completion ledger")
        with np.load(path, allow_pickle=False) as archive:
            grouped = {}
            for key in archive.files:
                parts = key.split("/")
                if len(parts) != 3:
                    raise ValueError(f"Invalid raw key {key}")
                layout, condition, metric = parts
                grouped.setdefault(layout, {}).setdefault(condition, {})[metric] = archive[key]
            if set(grouped) != set(indexed[str(path)]["layout_ids"]):
                raise ValueError("Shard layout content differs from index")
            for layout, raw in grouped.items():
                if layout in seen:
                    raise ValueError("Duplicate layout across shards")
                seen.add(layout)
                yield layout, raw


def validate_raw(raw, layout, arrays, native_probability):
    n = len(arrays["targets"])
    if set(raw) != {c["id"] for c in layout["conditions"]}:
        raise ValueError(f"{layout['id']}: condition set mismatch")
    targets = np.asarray(arrays["targets"])
    for name, metrics in raw.items():
        if set(metrics) != set(METRICS):
            raise ValueError(f"{name}: metric set mismatch")
        for metric, values in metrics.items():
            shape = (n, 16) if metric == "class_probability" else (n,)
            if np.asarray(values).shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name}/{metric}: invalid shape or nonfinite values")
        full = metrics["class_probability"]
        predicted = metrics["argmax_class"]
        if (np.any((full < 0) | (full > 1)) or not np.allclose(full.sum(1), 1, rtol=1e-6, atol=1e-7)
                or not np.array_equal(predicted, full.argmax(1))
                or not np.array_equal(metrics["accuracy"], predicted == targets)
                or not np.allclose(metrics["original_probability"], full[np.arange(n), targets], rtol=0, atol=1e-8)):
            raise ValueError(f"{name}: inconsistent full probabilities, argmax, target probability, or accuracy")
        if any(np.any(metrics[m] < 0) for m in STATE_METRICS):
            raise ValueError(f"{name}: negative absolute-error diagnostic")
        actual_error = np.abs(full-native_probability).max(1)
        if not np.allclose(metrics["class_probability_max_error_native"], actual_error, rtol=1e-5, atol=1e-7):
            raise ValueError(f"{name}: native probability diagnostic mismatch")


def primary_decisions(minimum_accuracy, deficit_ci, specificity_ci, protocol):
    gate = protocol["gates"]
    return {"every_layout_query_accuracy": minimum_accuracy >= gate["correct_each_layout_query_accuracy_min"],
            "native_minus_correct_upper": deficit_ci[1] <= gate["native_minus_correct_probability_ci95_upper_le"],
            "correct_minus_mean_shams_lower": specificity_ci[0] > gate["correct_minus_mean_shams_probability_ci95_lower_gt"]}


def score_seed(load_panel, arrays, grid, forecasts, protocol, weights, boundary_weights, emit=None):
    """Consume a panel iterator; emits CSV rows through emit(table, row)."""
    emit = emit or (lambda table, row: None)
    query = np.asarray(arrays["query_pair"])
    n = len(query)
    bn = boundary_weights.shape[1]
    boundary_arrays = {k: np.asarray(v)[:bn] for k, v in arrays.items()}
    bquery = query[:bn]
    if weights.shape[1] != n or set(query) != set(range(4)):
        raise ValueError("Weight/input mismatch")
    ref_specs = {grid["references"]["id"]: grid["references"]}
    references = list(load_panel("references"))
    if len(references) != 1 or references[0][0] not in ref_specs:
        raise ValueError("Missing or unexpected reference layout")
    reference_raw = references[0][1]
    native = reference_raw["native"]
    native_probability = native["class_probability"]
    maxima = {"noop_full_probability": 0., "correct_l1_value_native": 0., "correct_l1_query_native": 0.,
              "all_l1_key_baseline": 0., "all_l1_query_native": 0., "oracle_full_probability": 0.,
              "own_key_self_l1_value_reference": 0.}
    seen_counts = {}

    def consume(panel, specs, inputs, native_full, supplied=None):
        specs = {s["id"]: s for s in specs}
        seen = set()
        for layout_id, raw in (supplied if supplied is not None else load_panel(panel)):
            if layout_id not in specs or layout_id in seen:
                raise ValueError(f"Unexpected/duplicate layout {panel}/{layout_id}")
            seen.add(layout_id)
            layout = specs[layout_id]
            validate_raw(raw, layout, inputs, native_full)
            local_query = np.asarray(inputs["query_pair"])
            values = np.asarray(inputs["values"])
            for c in layout["conditions"]:
                name, metrics = c["id"], raw[c["id"]]
                maxima["all_l1_key_baseline"] = max(maxima["all_l1_key_baseline"], float(metrics["l1_key_max_error_baseline"].max()))
                maxima["all_l1_query_native"] = max(maxima["all_l1_query_native"], float(metrics["l1_query_max_error_native"].max()))
                base = {"panel": panel, "layout": layout_id, "condition": name, "oracle": bool(c.get("oracle"))}
                for q in [None, 0, 1, 2, 3]:
                    select = np.ones(len(local_query), dtype=bool) if q is None else local_query == q
                    accuracy = metrics["accuracy"][select]
                    bounds = wilson(accuracy)
                    probs = metrics["class_probability"][select]
                    displayed = probs[np.arange(len(probs))[:, None], values[select]]
                    missing = [] if q is None else layout["missing_native_predecessors"][q]
                    row = {**base, "query_pair": "all" if q is None else q, "n": len(accuracy),
                           "probability": float(metrics["original_probability"][select].mean()), "accuracy": float(accuracy.mean()),
                           "wilson_low": bounds[0], "wilson_high": bounds[1],
                           "missing_native_keys": "" if q is None else sum(t % 2 == 0 for t in missing),
                           "missing_native_values": "" if q is None else sum(t % 2 == 1 for t in missing),
                           **{f"displayed_value_{i}_probability": float(displayed[:, i].mean()) for i in range(4)}}
                    emit("condition_results" if q is None else "per_query_results", row)
            yield layout, raw
        if seen != set(specs):
            raise ValueError(f"Incomplete panel {panel}: {len(seen)} / {len(specs)}")
        seen_counts[panel] = len(seen)

    list(consume("references", [grid["references"]], arrays, native_probability, references))
    maxima["own_key_self_l1_value_reference"] = float(reference_raw["native_own_key_self"]["l1_value_max_error_own_key_self"].max())
    primary_prob, primary_accuracy = np.zeros(n), np.zeros(n)
    specificity = np.zeros(n)
    eligible = 0
    correct_cells = []
    grouped_correct = None
    grouped_reference = None
    for layout, raw in consume("primary", grid["primary"], arrays, native_probability):
        correct = raw["correct"]
        for index in np.flatnonzero(correct["accuracy"] == 0):
            emit("primary_failure_cases", {"layout": layout["id"], "row_index": int(index), "query_pair": int(query[index]),
                 "target": int(arrays["targets"][index]), "predicted": int(correct["argmax_class"][index]),
                 "target_probability": float(correct["original_probability"][index]),
                 "native_target_probability": float(native["original_probability"][index]),
                 "l1_key_max_error_native": float(correct["l1_key_max_error_native"][index])})
        primary_prob += correct["original_probability"]
        primary_accuracy += correct["accuracy"]
        maxima["noop_full_probability"] = max(maxima["noop_full_probability"], float(np.abs(raw["noop"]["class_probability"]-raw["unguarded"]["class_probability"]).max()))
        maxima["oracle_full_probability"] = max(maxima["oracle_full_probability"], float(np.abs(raw["oracle"]["class_probability"]-native_probability).max()))
        for part in ("value", "query"):
            key = f"correct_l1_{part}_native"
            maxima[key] = max(maxima[key], float(correct[f"l1_{part}_max_error_native"].max()))
        for q in range(4):
            selected = query == q
            row = {"layout": layout["id"], "query_pair": q, "accuracy": float(correct["accuracy"][selected].mean()),
                   "probability": float(correct["original_probability"][selected].mean()), "n": int(selected.sum())}
            correct_cells.append(row)
            if row["accuracy"] < protocol["gates"]["correct_each_layout_query_accuracy_min"]:
                emit("primary_failures", row)
        sham_names = [c["id"] for c in layout["conditions"] if c["id"].startswith("sham_")]
        if sham_names:
            sham = np.stack([raw[name]["original_probability"] for name in sham_names]).astype(np.float64)
            mean_sham = sham.mean(0)
            specificity += correct["original_probability"]-mean_sham
            eligible += 1
            best = int(sham.mean(1).argmax())
            emit("controls_summary", {"layout": layout["id"], "shams": len(sham_names),
                "correct_probability": float(correct["original_probability"].mean()), "mean_sham_probability": float(mean_sham.mean()),
                "best_sham": sham_names[best], "best_sham_probability": float(sham[best].mean()),
                "correct_minus_mean_shams": float((correct["original_probability"]-mean_sham).mean())})
        if layout["layout_indices"] == [0, 2, 4, 6, 1, 3, 5, 7, 8]:
            grouped_correct = correct["class_probability"].copy()
            grouped_reference = {name: {"probability": float(raw[name]["original_probability"][:bn].mean()),
                                        "accuracy": float(raw[name]["accuracy"][:bn].mean())}
                                 for name in ("unguarded", "correct")}
    if not eligible or grouped_correct is None:
        raise ValueError("Primary grid lacks eligible controls or grouped baseline")
    primary_prob /= len(grid["primary"])
    primary_accuracy /= len(grid["primary"])
    specificity /= eligible
    deficit = native["original_probability"].astype(np.float64)-primary_prob
    comparisons = {"native_minus_correct": mean_ci(deficit, weights), "correct_minus_mean_shams": mean_ci(specificity, weights),
                   "correct_probability": mean_ci(primary_prob, weights), "correct_accuracy": mean_ci(primary_accuracy, weights)}
    aggregate = {"native_mean_probability": float(native["original_probability"].mean()), "native_accuracy": float(native["accuracy"].mean()),
                 "primary_correct_mean_probability": float(primary_prob.mean()), "primary_correct_accuracy": float(primary_accuracy.mean()),
                 "primary_correct_min_layout_query_accuracy": min(row["accuracy"] for row in correct_cells),
                 "primary_native_minus_correct_probability": float(deficit.mean()),
                 "primary_correct_minus_mean_shams_probability": float(specificity.mean())}
    boundary_sums = {name: {"accuracy": np.zeros(bn), "probability": np.zeros(bn), "passing_cells": 0}
                     for name in ("unguarded", "keyguard", "logical_prefix", "own_key_self")}
    fixed_keys_sums = {name: {"accuracy": np.zeros(bn), "probability": np.zeros(bn)} for name in boundary_sums}
    fixed_keys_count = 0
    for layout, raw in consume("boundary", grid["boundary"], boundary_arrays, native_probability[:bn]):
        maxima["own_key_self_l1_value_reference"] = max(maxima["own_key_self_l1_value_reference"], float(raw["own_key_self"]["l1_value_max_error_own_key_self"].max()))
        for name, accumulator in boundary_sums.items():
            accumulator["accuracy"] += raw[name]["accuracy"]
            accumulator["probability"] += raw[name]["original_probability"]
            accumulator["passing_cells"] += sum(float(raw[name]["accuracy"][bquery == q].mean()) >= .95 for q in range(4))
        if layout["layout_indices"][:4] == [0, 2, 4, 6]:
            fixed_keys_count += 1
            for name, accumulator in fixed_keys_sums.items():
                accumulator["accuracy"] += raw[name]["accuracy"]
                accumulator["probability"] += raw[name]["original_probability"]
    boundary = {}
    for name, accumulator in boundary_sums.items():
        count = len(grid["boundary"])
        boundary[name] = {"accuracy": mean_ci(accumulator["accuracy"]/count, boundary_weights),
                          "probability": mean_ci(accumulator["probability"]/count, boundary_weights),
                          "fraction_layout_query_accuracy_ge_0_95": accumulator["passing_cells"]/(4*count)}
        if name in ("logical_prefix", "own_key_self"):
            aggregate[f"boundary_{name}_accuracy"] = boundary[name]["accuracy"]["mean"]
            aggregate[f"boundary_{name}_fraction_layout_query_accuracy_ge_0_95"] = boundary[name]["fraction_layout_query_accuracy_ge_0_95"]
    edge_values = np.zeros(n)
    edge_results = {}
    for layout, raw in consume("edge_panel", [grid["edge_panel"]], arrays, native_probability):
        values = np.asarray(arrays["values"])
        for c in layout["conditions"]:
            i, j = c["edge"]
            eligible_rows = query == j
            changes = raw[c["id"]]["class_probability"]-grouped_correct
            changes = changes[np.arange(n)[:, None], values]
            alternatives = [k for k in range(4) if k not in (i, j)]
            difference = changes[:, i]-changes[:, alternatives].mean(1)
            selected_weights = weights[:, eligible_rows]
            selected_weights = selected_weights/selected_weights.sum(1, keepdims=True)
            edge_results[c["id"]] = mean_ci(difference[eligible_rows], selected_weights)
            # Query-stratified bootstrap gives each stratum exactly 1/4 mass.
            edge_values[eligible_rows] += 4*difference[eligible_rows]/len(layout["conditions"])
    edge = mean_ci(edge_values, weights)
    aggregate["edge_wrong_value_difference_of_changes"] = edge["mean"]
    limits = protocol["gates"]
    validity = {"finite_complete_consistent": True, "native_accuracy": aggregate["native_accuracy"] >= limits["native_accuracy_min"],
                "noop_full_probability": maxima["noop_full_probability"] <= limits["noop_max_probability_error"],
                "correct_l1_value_native": maxima["correct_l1_value_native"] <= limits["identity_tolerance"],
                "correct_l1_query_native": maxima["correct_l1_query_native"] <= limits["identity_tolerance"],
                "all_l1_key_baseline": maxima["all_l1_key_baseline"] <= limits["identity_tolerance"],
                "all_l1_query_native": maxima["all_l1_query_native"] <= limits["identity_tolerance"],
                "oracle_full_probability": maxima["oracle_full_probability"] <= limits["identity_tolerance"],
                "own_key_self_l1_value_reference": maxima["own_key_self_l1_value_reference"] <= limits["identity_tolerance"]}
    primary = primary_decisions(aggregate["primary_correct_min_layout_query_accuracy"], comparisons["native_minus_correct"]["ci95"], comparisons["correct_minus_mean_shams"]["ci95"], protocol)
    secondary_limits = protocol["secondary_gates"]
    secondary = {"boundary_logical_prefix": aggregate["boundary_logical_prefix_accuracy"] >= secondary_limits["boundary_logical_prefix_accuracy_min"],
                 "boundary_own_key_self": aggregate["boundary_own_key_self_accuracy"] >= secondary_limits["boundary_own_key_self_accuracy_min"],
                 "edge_redirection": edge["ci95"][0] > secondary_limits["edge_wrong_value_difference_of_changes_ci95_lower_gt"]}
    if set(forecasts) != set(aggregate):
        raise ValueError("Forecast endpoint set mismatch")
    forecasts_out = []
    for endpoint, prediction in forecasts.items():
        error = aggregate[endpoint]-float(prediction)
        forecasts_out.append({"endpoint": endpoint, "forecast": prediction, "observed": aggregate[endpoint], "error": error,
                              "absolute_error": abs(error), "within_tolerance": abs(error) <= protocol["numeric_forecast_tolerance"]})
    fixed_keys = {"layouts": fixed_keys_count, "conditions": {name: {metric: mean_ci(values/fixed_keys_count, boundary_weights)
                 for metric, values in accumulator.items()} for name, accumulator in fixed_keys_sums.items()}} if fixed_keys_count else {"layouts": 0, "conditions": {}}
    return {"aggregate": aggregate, "comparisons": comparisons, "boundary": boundary, "boundary_fixed_key_order": fixed_keys,
            "grouped_reference_boundary_rows": grouped_reference, "edge": edge, "per_edge": edge_results,
            "validity_gates": validity, "primary_gates": primary, "secondary_gates": secondary, "manipulation_maxima": maxima,
            "primary_failed_cells": sum(row["accuracy"] < protocol["gates"]["correct_each_layout_query_accuracy_min"] for row in correct_cells),
            "worst_primary_cells": sorted(correct_cells, key=lambda r: (r["accuracy"], r["probability"]))[:20],
            "specificity_eligible_layouts": eligible, "panels_complete": seen_counts}, forecasts_out


def forecast_statistics(rows):
    errors = np.array([r["error"] for r in rows])
    return {"count": len(rows), "mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors**2))),
            "within_tolerance": sum(r["within_tolerance"] for r in rows), "fraction_within_tolerance": float(np.mean([r["within_tolerance"] for r in rows]))}


def main():
    protocol = json.loads(Path("experiment5/protocol.json").read_text())
    predictions = json.loads(Path("experiment5/predictions.json").read_text())
    grid = json.loads(Path("experiment5/conditions.json").read_text())
    if not (ROOT / "confirmatory/finish.json").is_file():
        raise SystemExit("Completed confirmatory finish record required before postprocessing")
    finish = json.loads((ROOT / "confirmatory/finish.json").read_text())
    if finish["seeds"] != protocol["model_seeds"] or finish["condition_counts"] != grid["counts"]:
        raise ValueError("Finish record differs from registered model/condition grid")
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    n, bn = protocol["n"], protocol["boundary_n"]
    if len(arrays["targets"]) != n or not np.array_equal(np.bincount(arrays["query_pair"], minlength=4), [n//4]*4) or not np.array_equal(np.bincount(arrays["query_pair"][:bn], minlength=4), [bn//4]*4):
        raise ValueError("Registered balanced input counts violated")
    boot = protocol["bootstrap"]
    weights = stratified_weights(arrays["query_pair"], boot["replicates"], boot["seed"])
    bweights = stratified_weights(arrays["query_pair"][:bn], boot["replicates"], boot["boundary_seed"])
    scores, forecast_rows, writers, handles = {}, [], {}, []
    try:
        for seed in protocol["model_seeds"]:
            def emit(table, row):
                row = {"seed": seed, **row}
                if table not in writers:
                    handle = (ROOT / f"{table}.csv").open("w", newline="")
                    handles.append(handle)
                    writers[table] = csv.DictWriter(handle, fieldnames=list(row))
                    writers[table].writeheader()
                writers[table].writerow(row)
            score, rows = score_seed(lambda panel: iter_saved_panel(seed, panel), arrays, grid,
                                     predictions["expected_aggregate"][str(seed)], protocol, weights, bweights, emit)
            scores[str(seed)] = score
            for row in rows:
                forecast_rows.append({"seed": seed, **row})
                emit("forecasts_vs_results", row)
                if not row["within_tolerance"]:
                    emit("forecast_misses", row)
            print(json.dumps({"seed": seed, "validity": score["validity_gates"], "primary": score["primary_gates"], "secondary": score["secondary_gates"]}), flush=True)
    finally:
        for handle in handles:
            handle.close()
    # Ensure empty failure tables exist instead of leaving ambiguity or stale files.
    for table, fields in (("primary_failures", ["seed", "layout", "query_pair", "accuracy", "probability", "n"]),
                          ("primary_failure_cases", ["seed", "layout", "row_index", "query_pair", "target", "predicted", "target_probability", "native_target_probability", "l1_key_max_error_native"]),
                          ("forecast_misses", ["seed", "endpoint", "forecast", "observed", "error", "absolute_error", "within_tolerance"])):
        if table not in writers:
            with (ROOT / f"{table}.csv").open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
    decisions = {"all_six_valid": all(all(s["validity_gates"].values()) for s in scores.values()),
                 "all_six_recovery": all(s["primary_gates"]["every_layout_query_accuracy"] and s["primary_gates"]["native_minus_correct_upper"] for s in scores.values()),
                 "all_six_specificity": all(s["primary_gates"]["correct_minus_mean_shams_lower"] for s in scores.values())}
    decisions["primary_claim_pass"] = all(decisions.values())
    decisions["secondary"] = {name: decisions["all_six_valid"] and all(s["secondary_gates"][name] for s in scores.values()) for name in ("boundary_logical_prefix", "boundary_own_key_self", "edge_redirection")}
    for filename, value in (("scores", scores), ("global_decisions", decisions), ("forecast_statistics", forecast_statistics(forecast_rows))):
        (ROOT / f"confirmatory/{filename}.json").write_text(json.dumps(value, indent=2)+"\n")


if __name__ == "__main__":
    main()
