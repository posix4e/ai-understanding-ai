"""Frozen E6 postprocessing: saved arrays only, no model or calibration imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("native", "grouped_physical", "grouped_canonical", "grouped_noop",
              "guard_block0", *(f"sham_{i}" for i in range(8)), "guard_block5", "guard_all")
FIELDS = {"candidate_probs", "top_token_ids", "target_token_ids", "target_indices", "query_pair"}
BOOTSTRAP_SEED = 10600003
BOOTSTRAP_DRAWS = 2000


def stratified_weights(query, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    query = np.asarray(query)
    if query.ndim != 1 or not np.isin(query, range(4)).all():
        raise ValueError("query_pair must contain four integer strata")
    sizes = [int(np.sum(query == q)) for q in range(4)]
    if not sizes[0] or len(set(sizes)) != 1:
        raise ValueError("Exactly balanced nonempty query strata required")
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(query)), dtype=np.float64)
    for q, size in enumerate(sizes):
        indices = np.flatnonzero(query == q)
        weights[:, indices] = rng.multinomial(size, np.full(size, 1 / size), size=draws) / len(query)
    return weights


def interval(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "ci95": np.quantile(weights @ values, [.025, .975]).tolist()}


def wilson(values):
    values = np.asarray(values, dtype=float)
    n, p, z = len(values), float(values.mean()), 1.959963984540054
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0., float(center-radius)), min(1., float(center+radius))]


def decisions(native_accuracy, native_strata, damage, repair, deficit, specificity, valid):
    gates = {
        "implementation_valid": bool(valid),
        "native_overall_at_least_80_percent": native_accuracy >= .80,
        "every_native_query_at_least_70_percent": min(native_strata) >= .70,
        "damage_point_at_least_10_percentage_points": damage["mean"] >= .10,
        "damage_ci_lower_above_zero": damage["ci95"][0] > 0.,
        "repair_accuracy_ci_lower_above_5_percentage_points": repair["ci95"][0] > .05,
        "native_minus_guard_accuracy_ci_upper_at_most_5_percentage_points": deficit["ci95"][1] <= .05,
        "specificity_conditional_probability_ci_lower_above_02": specificity["ci95"][0] > .02,
    }
    if not gates["implementation_valid"]:
        classification = "implementation_invalid"
    elif not all(list(gates.values())[1:3]):
        classification = "task_invalid"
    elif not all(list(gates.values())[3:5]):
        classification = "no_damage"
    elif not all(gates.values()):
        classification = "repair_failed"
    else:
        classification = "all_pass"
    return {"gates": gates, "failed_gates": [k for k, v in gates.items() if not v],
            "classification": classification, "all_pass": all(gates.values())}


def validate_raw(raw, candidate_token_ids=None, expected_n=None):
    if set(raw) != set(CONDITIONS):
        raise ValueError("Expected exactly the 15 registered conditions")
    ref = raw["native"]
    n = len(ref["query_pair"])
    if expected_n is not None and n != expected_n:
        raise ValueError("Sample size differs from frozen protocol")
    stratified_weights(ref["query_pair"], draws=1)
    candidates = None if candidate_token_ids is None else np.asarray(candidate_token_ids)
    if candidates is not None and (candidates.shape != (16,) or len(set(candidates.tolist())) != 16
                                   or candidates.dtype.kind not in "iu" or np.any(candidates < 0)):
        raise ValueError("Invalid fixed candidate-token mapping")
    for name, arrays in raw.items():
        if set(arrays) != FIELDS:
            raise ValueError(f"{name}: unexpected saved fields")
        for field, array in arrays.items():
            array = np.asarray(array)
            shape = (n, 16) if field == "candidate_probs" else (n,)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name}/{field}: invalid shape or nonfinite values")
            if field != "candidate_probs" and (array.dtype.kind not in "iu" or np.any(array < 0)):
                raise ValueError(f"{name}/{field}: expected nonnegative integer IDs")
        probs = np.asarray(arrays["candidate_probs"], dtype=np.float64)
        if np.any((probs < 0) | (probs > 1)) or np.any(probs.sum(1) > 1+1e-6) or np.any(probs.sum(1) <= 0):
            raise ValueError(f"{name}: invalid raw candidate probabilities or zero candidate mass")
        targets = arrays["target_indices"]
        if np.any(targets >= 16):
            raise ValueError(f"{name}: target index outside 16 candidates")
        for field in ("target_indices", "target_token_ids", "query_pair"):
            if not np.array_equal(arrays[field], ref[field]):
                raise ValueError(f"{name}: row metadata differs across conditions")
        if candidates is not None and not np.array_equal(candidates[targets], arrays["target_token_ids"]):
            raise ValueError(f"{name}: inconsistent target-token mapping")
        if candidates is not None:
            # A full-vocabulary winner that is a candidate must also maximize candidate probability.
            for row, token in enumerate(arrays["top_token_ids"]):
                hit = np.flatnonzero(candidates == token)
                if len(hit) and probs[row, hit[0]] < probs[row].max():
                    raise ValueError(f"{name}: full-vocabulary winner contradicts candidate probabilities")
    return n


def score_conditions(raw, validity, candidate_token_ids=None, expected_n=None):
    n = validate_raw(raw, candidate_token_ids, expected_n)
    checks = validity.get("checks")
    if (not isinstance(checks, dict) or not checks or any(type(v) is not bool for v in checks.values())
            or type(validity.get("all_pass")) is not bool or validity["all_pass"] != all(checks.values())):
        raise ValueError("Invalid runner validity record; all_pass must equal conjunction of boolean checks")
    query = raw["native"]["query_pair"]
    weights = stratified_weights(query)
    vectors, summaries = {}, {}
    rows = np.arange(n)
    for name in CONDITIONS:
        arrays = raw[name]
        probs = np.asarray(arrays["candidate_probs"], dtype=np.float64)
        mass = probs.sum(1)
        target = probs[rows, arrays["target_indices"]]
        metrics = {
            "restricted_accuracy": (probs.argmax(1) == arrays["target_indices"]).astype(float),
            "unrestricted_accuracy": (arrays["top_token_ids"] == arrays["target_token_ids"]).astype(float),
            "raw_target_probability": target,
            "conditional_target_probability": target / mass,
            "candidate_probability_mass": mass,
        }
        vectors[name] = metrics
        summary = {metric: interval(v, weights) for metric, v in metrics.items()}
        summary["candidate_argmax_tie_rows"] = int(np.sum((probs == probs.max(1, keepdims=True)).sum(1) > 1))
        summary["per_query"] = {}
        for q in range(4):
            select = query == q
            summary["per_query"][str(q)] = {"n": int(select.sum()), **{
                metric: {"mean": float(v[select].mean()), **({"wilson_ci95": wilson(v[select])}
                         if "accuracy" in metric else {})} for metric, v in metrics.items()}}
        summaries[name] = summary
    native = vectors["native"]["restricted_accuracy"]
    grouped = vectors["grouped_canonical"]["restricted_accuracy"]
    guard = vectors["guard_block0"]["restricted_accuracy"]
    sham_conditional = np.stack([vectors[f"sham_{i}"]["conditional_target_probability"] for i in range(8)])
    guard_conditional = vectors["guard_block0"]["conditional_target_probability"]
    contrasts = {
        "damage_native_minus_grouped_accuracy": interval(native-grouped, weights),
        "repair_guard_minus_grouped_accuracy": interval(guard-grouped, weights),
        "deficit_native_minus_guard_accuracy": interval(native-guard, weights),
        "specificity_guard_minus_mean_shams_conditional_probability": interval(guard_conditional-sham_conditional.mean(0), weights),
        "guard_minus_grouped_raw_target_probability": interval(vectors["guard_block0"]["raw_target_probability"]-vectors["grouped_canonical"]["raw_target_probability"], weights),
        "guard_minus_grouped_candidate_mass": interval(vectors["guard_block0"]["candidate_probability_mass"]-vectors["grouped_canonical"]["candidate_probability_mass"], weights),
    }
    best_index = int(np.argmax(sham_conditional.mean(1)))
    contrasts["guard_minus_best_point_sham_conditional_probability_descriptive"] = {
        "sham": f"sham_{best_index}", "mean": float((guard_conditional-sham_conditional[best_index]).mean())}
    for name in ("guard_block5", "guard_all"):
        contrasts[f"secondary_{name}_minus_grouped_accuracy"] = interval(vectors[name]["restricted_accuracy"]-grouped, weights)
    result = decisions(float(native.mean()), [float(native[query == q].mean()) for q in range(4)],
                       contrasts["damage_native_minus_grouped_accuracy"], contrasts["repair_guard_minus_grouped_accuracy"],
                       contrasts["deficit_native_minus_guard_accuracy"],
                       contrasts["specificity_guard_minus_mean_shams_conditional_probability"], validity["all_pass"])
    result.update({"n": n, "conditions": summaries, "contrasts": contrasts, "runner_validity": validity,
                   "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "dictionary row",
                                 "stratified_by": "query_pair", "shared_across_conditions": True},
                   "scoring": "16-choice argmax; ties resolved by first index in frozen candidate order",
                   "interval_scope": "Individual paired percentile intervals; not simultaneous intervals."})
    return result


def validate_input_labels(raw, cases, values):
    """Check all saved row labels against the frozen input dictionaries."""
    native = raw["native"]
    if len(cases) != len(native["query_pair"]) or len(values) != 16 or len(set(values)) != 16:
        raise ValueError("Frozen input length or value vocabulary mismatch")
    query = np.asarray([case["query_pair"] for case in cases])
    indices = np.asarray([values.index(case["values"][case["query_pair"]]) for case in cases])
    if not np.array_equal(query, native["query_pair"]) or not np.array_equal(indices, native["target_indices"]):
        raise ValueError("Saved outcome labels differ from frozen inputs")


def render_markdown(scores):
    interpretations = {
        "implementation_invalid": "Implementation checks failed. Behavioral numbers cannot validate the repair claim.",
        "task_invalid": "GPT-2 did not meet the native lookup competence criterion. This is not a successful transfer test.",
        "no_damage": "The registered ordering-damage criterion was not met. The proposed failure requiring repair was not established.",
        "repair_failed": "Native competence and ordering damage were established, but the primary repair did not satisfy every registered criterion.",
        "all_pass": "The fixed first-block intervention satisfied the registered repair criteria on this GPT-2 checkpoint and calibrated task format.",
    }
    lines = ["# E6: pretrained GPT-2 confirmation", "", f"**Classification: {scores['classification']}.** " + interpretations[scores['classification']], "",
             f"{scores['n']} fresh dictionaries; one pretrained checkpoint. Restricted accuracy ranks 16 answer tokens and is not unrestricted generation accuracy.", "",
             "| Condition | Restricted accuracy | Unrestricted next-token accuracy | Raw target probability | Conditional target probability | Candidate mass |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in scores["conditions"].items():
        vals = [record[k]["mean"] for k in ("restricted_accuracy", "unrestricted_accuracy", "raw_target_probability", "conditional_target_probability", "candidate_probability_mass")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    lines += ["", "## Registered gates", "", "| Gate | Outcome |", "| --- | --- |"]
    lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in scores["gates"].items()]
    lines += ["", "## Paired contrasts", "", "| Contrast | Mean | Individual 95% interval |", "| --- | ---: | --- |"]
    for name, record in scores["contrasts"].items():
        bounds = record.get("ci95")
        ci = f"[{bounds[0]:.6f}, {bounds[1]:.6f}]" if bounds else "Descriptive; selected by point estimate"
        lines.append(f"| {name} | {record['mean']:.6f} | {ci} |")
    lines += ["", "## Query-position results", "", "| Condition | Query position | N | Restricted accuracy | Unrestricted accuracy | Conditional target probability |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in scores["conditions"].items():
        for q, row in record["per_query"].items():
            lines.append(f"| {name} | {q} | {row['n']} | {row['restricted_accuracy']['mean']:.6f} | {row['unrestricted_accuracy']['mean']:.6f} | {row['conditional_target_probability']['mean']:.6f} |")
    lines += ["", "Intervals use 2,000 paired query-stratified bootstrap draws shared across all conditions (seed 10600003). They are individual intervals, not simultaneous confidence coverage. Per-query Wilson intervals and all saved metrics are in scores.json.", "",
              "Middle-block and all-block interventions are secondary and cannot rescue a failed primary claim. First-layer state identities are implementation checks, not evidence of final-answer recovery through 12 layers. The mask uses externally supplied pair relationships and position labels; success would not establish spontaneous layout generalization, a unique algorithm, or generalization to other pretrained models.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/experiment6"))
    parser.add_argument("--protocol", type=Path, default=Path("experiment6/protocol.json"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    directory = args.root / "confirmatory"
    paths = {name: directory / f"{name}.npz" for name in CONDITIONS}
    if {p.stem for p in directory.glob("*.npz")} != set(CONDITIONS):
        raise ValueError("Missing or extra raw condition files")
    raw = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            raw[name] = {field: archive[field] for field in archive.files}
    validity = json.loads((directory / "validity.json").read_text())
    if protocol["n"] not in (256, 512):
        raise ValueError("Confirmation sample size must be 256 or 512")
    scores = score_conditions(raw, validity, protocol["candidate_token_ids"], protocol["n"])
    input_path = Path(protocol["inputs_path"])
    inputs = json.loads(input_path.read_text())
    validate_input_labels(raw, inputs, protocol["values"])
    scores["raw_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    scores["protocol_sha256"] = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    scores["inputs_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    (directory / "scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
    (args.root / "RESULTS.md").write_text(render_markdown(scores))
    print(json.dumps({k: scores[k] for k in ("classification", "all_pass", "failed_gates")}, indent=2))


if __name__ == "__main__":
    main()
