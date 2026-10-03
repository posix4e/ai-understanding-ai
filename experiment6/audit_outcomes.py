"""Independent POST-REGISTRATION audit of completed E6 saved outputs.

This tool was written after registration, is not part of the frozen analysis,
and never loads or evaluates a model. It imports no experiment implementation.
State identities and full-vocabulary no-op errors cannot be reconstructed from
the saved candidate probabilities; those remain explicitly runner-reported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


METRICS = ("restricted_accuracy", "unrestricted_accuracy", "raw_target_probability",
           "conditional_target_probability", "candidate_probability_mass")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def make_bootstrap_counts(query, draws, seed):
    """Independently form integer dictionary multiplicities, shared by all scores."""
    generator = np.random.default_rng(seed)
    counts = np.zeros((draws, len(query)), dtype=np.int64)
    for label in range(4):
        rows = np.where(query == label)[0]
        require(len(rows) * 4 == len(query), "queries are not exactly balanced")
        counts[:, rows] = generator.multinomial(len(rows), np.repeat(1 / len(rows), len(rows)), size=draws)
    require(np.all(counts.sum(axis=1) == len(query)), "bootstrap sample sizes changed")
    return counts


def estimate(vector, counts):
    vector = np.asarray(vector, dtype=np.float64)
    draws = (counts @ vector) / len(vector)
    return {"mean": float(np.sum(vector) / len(vector)),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}


def binomial_interval(vector):
    n = len(vector)
    p = float(np.sum(vector) / n)
    z2 = 1.959963984540054**2
    mid = (n * p + z2 / 2) / (n + z2)
    half = np.sqrt(z2 * (n * p * (1-p) + z2 / 4)) / (n + z2)
    return [float(max(0, mid-half)), float(min(1, mid+half))]


def audit(repository, directory, registration, protocol_path=None, manifest_path=None):
    protocol_path = repository / (protocol_path or "experiment6/protocol.json")
    manifest_path = repository / (manifest_path or "outputs/experiment6/confirmation_manifest.json")
    result = {"audit_kind": "independent_saved_output_audit_after_registration",
              "registration": registration, "model_forwards": 0,
              "official_analysis_imports": False,
              "audit_tool_sha256": digest(__file__),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    manifest = read_json(manifest_path)

    def committed_bytes(name):
        return subprocess.check_output(["git", "show", f"{registration}:{name}"], cwd=repository)

    require(manifest_path.read_bytes() == committed_bytes(str(manifest_path.relative_to(repository))),
            "manifest differs from the registered manifest")
    registered_hashes = {}
    for name, expected in manifest["files"].items():
        path = repository / name
        actual = digest(path)
        require(actual == expected, f"registered hash changed: {name}")
        require(path.read_bytes() == committed_bytes(name), f"registered bytes changed: {name}")
        registered_hashes[name] = actual
    essential = {"experiment6/protocol.json", "experiment6/PROTOCOL.md", "experiment6/score.py",
                 "experiment6/run_phase.py", "experiment6/data.py", "experiment6/model_adapter.py",
                 "experiment6/conditions.py", "experiment6/requirements.txt", "experiment6/predictions.json",
                 "outputs/experiment6/confirmation_inputs.json", "outputs/experiment6/model_manifest.json"}
    essential.add(str(protocol_path.relative_to(repository)))
    require(essential.issubset(registered_hashes), "registration omits a required scientific input")
    result["registered_files"] = {"count": len(registered_hashes), "all_unchanged": True,
                                  "sha256": registered_hashes}

    protocol = read_json(protocol_path)
    cases = read_json(repository / protocol["inputs_path"])
    candidates = np.asarray(protocol["candidate_token_ids"], dtype=np.int64)
    values = protocol["values"]
    n = protocol["n"]
    conditions = [c["id"] for c in protocol["conditions"]]
    require(len(conditions) == 15 and len(set(conditions)) == 15, "condition grid changed")
    require(n == 512 and len(cases) == n, "expected 512 frozen dictionaries")
    require(len(values) == 16 and len(set(values)) == 16 and candidates.shape == (16,), "invalid answer vocabulary")
    require(protocol["bootstrap_seed"] == 10600003 and protocol["bootstrap_draws"] == 2000,
            "bootstrap differs from registered plan")
    model_manifest = read_json(repository / "outputs/experiment6/model_manifest.json")
    require(model_manifest["revision"] == protocol["revision"], "model revision differs")
    require(model_manifest["model_id"] == protocol["model_id"], "model identity differs")
    model_root = repository / "work/models/gpt2"
    for name, record in model_manifest["files"].items():
        require(digest(model_root / name) == record["sha256"], f"model/tokenizer file changed: {name}")
        require((model_root / name).stat().st_size == record["bytes"], f"model file size changed: {name}")
    vocabulary = read_json(model_root / "vocab.json")
    require([vocabulary["\u0120" + str(value)] for value in values] == candidates.tolist(),
            "answer token IDs disagree with the frozen GPT-2 vocabulary")
    vocab_size = read_json(model_root / "config.json")["vocab_size"]
    query = np.array([case["query_pair"] for case in cases], dtype=np.int64)
    target_index = np.array([values.index(case["values"][case["query_pair"]]) for case in cases], dtype=np.int64)
    target_token = candidates[target_index]
    require(np.isin(query, range(4)).all(), "invalid query label")
    maps = [tuple(sorted(zip(case["keys"], case["values"]))) for case in cases]
    calibration = read_json(repository / "outputs/experiment6/calibration_inputs.json")
    calibration_maps = {tuple(sorted(zip(case["keys"], case["values"]))) for case in calibration}
    require(len(set(maps)) == n and not set(maps).intersection(calibration_maps), "duplicate or calibration dictionary")
    result["model_and_labels"] = {"model_files_verified": len(model_manifest["files"]),
                                  "answer_ids_match_vocabulary": True, "labels_from_frozen_inputs": True,
                                  "unique_maps": n, "calibration_overlap": 0}

    started = read_json(directory / "started.json")
    finished = read_json(directory / "finished.json")
    require(started["registration"] == finished["registration"] == registration, "run registration differs")
    require(started["n"] == finished["n"] == n, "completion row count differs")
    require(started["conditions"] == finished["conditions"] == len(conditions), "completion condition count differs")
    require(started["dtype"] == protocol["dtype"], "recorded computation precision differs from protocol")
    require(datetime.fromisoformat(finished["at_utc"]) >= datetime.fromisoformat(started["at_utc"]), "reversed timestamps")
    require({p.stem for p in directory.glob("*.npz")} == set(conditions), "merged files differ from grid")
    require(set(finished["files"]) == {c + ".npz" for c in conditions}, "completion hash list differs")
    fields = {"candidate_probs", "top_token_ids", "target_indices", "target_token_ids", "query_pair"}
    raw = {}
    merged_hashes = {}
    for name in conditions:
        path = directory / (name + ".npz")
        merged_hashes[name] = digest(path)
        require(merged_hashes[name] == finished["files"][path.name], f"merged hash mismatch: {name}")
        arrays = load_arrays(path)
        require(set(arrays) == fields, f"merged fields differ: {name}")
        for field, array in arrays.items():
            require(array.shape == ((n,16) if field == "candidate_probs" else (n,)), f"shape mismatch: {name}/{field}")
            require(np.isfinite(array).all(), f"nonfinite values: {name}/{field}")
            if field != "candidate_probs":
                require(array.dtype.kind in "iu", f"noninteger IDs: {name}/{field}")
        for field, expected in (("target_indices",target_index),("target_token_ids",target_token),("query_pair",query)):
            require(np.array_equal(arrays[field], expected), f"input label mismatch: {name}/{field}")
        p = arrays["candidate_probs"].astype(np.float64)
        mass = p.sum(axis=1)
        require(np.all((p >= 0) & (p <= 1)) and np.all((mass > 0) & (mass <= 1+1e-6)), f"invalid probability mass: {name}")
        require(np.all((arrays["top_token_ids"] >= 0) & (arrays["top_token_ids"] < vocab_size)), f"top token outside vocabulary: {name}")
        for row, token in enumerate(arrays["top_token_ids"]):
            matches = np.where(candidates == token)[0]
            require(not len(matches) or p[row,matches[0]] == p[row].max(), f"inconsistent candidate/full winner: {name}/{row}")
        raw[name] = arrays

    ledger = read_json(directory / "ledger.json")
    batch_size = protocol["batch_size"]
    require(len(ledger) == (n+batch_size-1)//batch_size == 64, "expected all 64 batch records")
    expected_chunk_fields = {f"{name}__{field}" for name in conditions for field in ("candidate_probs","top_token_ids")} | {"case_indices"}
    seen = np.zeros(n, dtype=np.int64)
    paths = []
    for record in ledger:
        path = (directory / record["file"]).resolve()
        require(path.is_relative_to((directory / "batches").resolve()), "batch escapes its output directory")
        require(digest(path) == record["sha256"], f"batch hash mismatch: {path.name}")
        arrays = load_arrays(path)
        require(set(arrays) == expected_chunk_fields, f"batch field mismatch: {path.name}")
        rows = np.arange(record["begin"],record["end"])
        require(0 <= record["begin"] < record["end"] <= n and len(rows) == batch_size, "invalid batch row bounds")
        require(np.array_equal(arrays["case_indices"], rows), "batch indices disagree with ledger")
        seen[rows] += 1
        for name in conditions:
            for field in ("candidate_probs","top_token_ids"):
                require(np.array_equal(arrays[f"{name}__{field}"],raw[name][field][rows]), f"batch/merged mismatch: {name}/{field}/{record['begin']}")
        paths.append(path)
    require(np.all(seen == 1) and len(set(paths)) == 64, "batch coverage has missing/duplicate rows")
    require(set(paths) == {p.resolve() for p in (directory / "batches").glob("*.npz")}, "unlisted batch files")
    result["saved_arrays"] = {"merged_conditions": 15, "batch_files": 64, "rows_per_condition": n,
                              "all_batch_hashes_match": True, "all_merged_hashes_match": True,
                              "batch_arrays_equal_merged_arrays": True, "all_rows_covered_once": True,
                              "all_probabilities_finite_and_valid": True}

    validity = read_json(directory / "validity.json")
    diagnostic = validity["diagnostics"]
    expected_validity = {
        "complete_finite_outputs": True,
        "parameters_unchanged": validity["parameters_before_sha256"] == validity["parameters_after_sha256"],
        "noop_probabilities": diagnostic["noop_probability_error"] <= protocol["noop_probability_tolerance"],
        "guard_first_block_values": diagnostic["guard_value_state_error"] <= protocol["state_tolerance"],
        "first_block_query_suffix": diagnostic["query_suffix_state_error"] <= protocol["state_tolerance"],
        "unedited_first_block_keys": diagnostic["unedited_key_state_error"] <= protocol["state_tolerance"],
    }
    if "source_parameter_sha256" in protocol:
        require(validity["source_float32_parameter_sha256"] == protocol["source_parameter_sha256"],
                "promoted model's reported source hash differs from protocol")
        source_started = read_json(repository / "outputs/experiment6/confirmatory/started.json")
        require(source_started["registration"] == protocol["source_registration"], "source run registration differs")
        require(source_started["parameters_sha256"] == protocol["source_parameter_sha256"],
                "source parameter hash differs from original run")
        require(started["source_parameter_dtype"] == protocol["source_parameter_dtype"] == "float32"
                and started["dtype"] == "float64", "invalid precision-promotion metadata")
        reported_preservation = validity["checks"].get("source_parameter_values_preserved")
        require(type(reported_preservation) is bool, "missing source-value preservation measurement")
        expected_validity["source_parameter_values_preserved"] = reported_preservation
    require(all(np.isfinite(x) and x >= 0 for x in diagnostic.values()), "invalid reported state diagnostics")
    require(validity["parameters_before_sha256"] == started["parameters_sha256"], "initial parameter hash differs across records")
    require(validity["checks"] == expected_validity, "runner validity flags contradict its measurements")
    implementation_valid = all(expected_validity.values())
    require(validity["all_pass"] == finished["validity_pass"] == implementation_valid, "validity conjunction mismatch")
    candidate_noop_error = float(np.max(np.abs(raw["grouped_noop"]["candidate_probs"] - raw["grouped_canonical"]["candidate_probs"])))
    require(candidate_noop_error <= diagnostic["noop_probability_error"] + 1e-12, "candidate no-op error exceeds reported full-vocabulary error")
    result["implementation_evidence_scope"] = {
        "independent_candidate_noop_max_error": candidate_noop_error,
        "runner_reported_validity_flags_consistent": True,
        "full_states_reconstructed": False, "full_vocabulary_noop_reconstructed": False,
        "runtime_parameter_hashes_independently_reconstructed": False,
        "source_value_preservation_independently_reconstructed": False,
        "limitation": "State identities, full-vocabulary no-op and runtime parameter hashes are runner-reported measurements, not independently recoverable from saved candidate probabilities. Frozen checkpoint file hashes were independently checked."}

    counts = make_bootstrap_counts(query, protocol["bootstrap_draws"], protocol["bootstrap_seed"])
    vectors, summaries = {}, {}
    for name, arrays in raw.items():
        p = arrays["candidate_probs"].astype(np.float64)
        target = p[np.arange(n),target_index]
        mass = p.sum(axis=1)
        vectors[name] = {
            "restricted_accuracy": (p.argmax(axis=1) == target_index).astype(np.float64),
            "unrestricted_accuracy": (arrays["top_token_ids"] == target_token).astype(np.float64),
            "raw_target_probability": target, "conditional_target_probability": target / mass,
            "candidate_probability_mass": mass,
        }
        summaries[name] = {metric: estimate(vector,counts) for metric,vector in vectors[name].items()}
        summaries[name]["candidate_argmax_tie_rows"] = int(np.sum(np.sum(p == p.max(axis=1,keepdims=True),axis=1)>1))
        summaries[name]["per_query"] = {}
        for label in range(4):
            selected = query == label
            summaries[name]["per_query"][str(label)] = {"n": int(selected.sum())}
            for metric,vector in vectors[name].items():
                item = {"mean":float(vector[selected].mean())}
                if metric in ("restricted_accuracy","unrestricted_accuracy"):
                    item["wilson_ci95"] = binomial_interval(vector[selected])
                summaries[name]["per_query"][str(label)][metric] = item
    acc = {name: vectors[name]["restricted_accuracy"] for name in conditions}
    conditional = {name: vectors[name]["conditional_target_probability"] for name in conditions}
    shams = np.stack([conditional[f"sham_{i}"] for i in range(8)])
    contrasts = {
        "damage_native_minus_grouped_accuracy": estimate(acc["native"]-acc["grouped_canonical"],counts),
        "repair_guard_minus_grouped_accuracy": estimate(acc["guard_block0"]-acc["grouped_canonical"],counts),
        "deficit_native_minus_guard_accuracy": estimate(acc["native"]-acc["guard_block0"],counts),
        "specificity_guard_minus_mean_shams_conditional_probability": estimate(conditional["guard_block0"]-shams.mean(axis=0),counts),
    }
    for metric,suffix in (("raw_target_probability","raw_target_probability"),("candidate_probability_mass","candidate_mass")):
        contrasts[f"guard_minus_grouped_{suffix}"] = estimate(vectors["guard_block0"][metric]-vectors["grouped_canonical"][metric],counts)
    best_sham = int(np.argmax(shams.mean(axis=1)))
    contrasts["guard_minus_best_point_sham_conditional_probability_descriptive"] = {
        "sham": f"sham_{best_sham}", "mean": float((conditional["guard_block0"]-shams[best_sham]).mean())}
    for name in ("guard_block5","guard_all"):
        contrasts[f"secondary_{name}_minus_grouped_accuracy"] = estimate(acc[name]-acc["grouped_canonical"],counts)
    damage = contrasts["damage_native_minus_grouped_accuracy"]
    repair = contrasts["repair_guard_minus_grouped_accuracy"]
    deficit = contrasts["deficit_native_minus_guard_accuracy"]
    specificity = contrasts["specificity_guard_minus_mean_shams_conditional_probability"]
    gates = {
        "implementation_valid": implementation_valid,
        "native_overall_at_least_80_percent": bool(acc["native"].mean() >= .80),
        "every_native_query_at_least_70_percent": all(acc["native"][query==q].mean() >= .70 for q in range(4)),
        "damage_point_at_least_10_percentage_points": damage["mean"] >= .10,
        "damage_ci_lower_above_zero": damage["ci95"][0] > 0,
        "repair_accuracy_ci_lower_above_5_percentage_points": repair["ci95"][0] > .05,
        "native_minus_guard_accuracy_ci_upper_at_most_5_percentage_points": deficit["ci95"][1] <= .05,
        "specificity_conditional_probability_ci_lower_above_02": specificity["ci95"][0] > .02,
    }
    if not implementation_valid:
        classification = "implementation_invalid"
    elif not (gates["native_overall_at_least_80_percent"] and gates["every_native_query_at_least_70_percent"]):
        classification = "task_invalid"
    elif not (gates["damage_point_at_least_10_percentage_points"] and gates["damage_ci_lower_above_zero"]):
        classification = "no_damage"
    elif all(gates.values()):
        classification = "all_pass"
    else:
        classification = "repair_failed"

    official_path = directory / "scores.json"
    official = read_json(official_path)
    comparison = {"numeric_scalars":0,"max_absolute_difference":0.,"absolute_tolerance":1e-10}
    def compare(expected, actual, location):
        if isinstance(expected,dict):
            for key,value in expected.items():
                require(key in actual, f"official score lacks {location}/{key}")
                compare(value,actual[key],location+"/"+key)
        elif isinstance(expected,list):
            require(len(actual)==len(expected),f"official list length differs: {location}")
            for index,value in enumerate(expected):
                compare(value,actual[index],location+f"/{index}")
        elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
            error=abs(float(expected)-float(actual))
            require(np.isfinite(error) and error<=comparison["absolute_tolerance"],f"official numerical mismatch: {location}: {expected} vs {actual}")
            comparison["numeric_scalars"]+=1
            comparison["max_absolute_difference"]=max(comparison["max_absolute_difference"],error)
        else:
            require(expected==actual,f"official result mismatch: {location}")
    compare(summaries,official["conditions"],"conditions")
    compare(contrasts,official["contrasts"],"contrasts")
    compare(gates,official["gates"],"gates")
    require(official["classification"]==classification and official["all_pass"]==all(gates.values()),"official classification differs")
    require(official["failed_gates"]==[name for name,passed in gates.items() if not passed],"failed gates differ")
    require(official["n"]==n,"official sample size differs")
    require(official["raw_sha256"]==merged_hashes,"official raw hashes differ")
    require(official["protocol_sha256"]==digest(protocol_path),"official protocol hash differs")
    require(official["inputs_sha256"]==digest(repository/protocol['inputs_path']),"official inputs hash differs")
    require(official["runner_validity"]==validity,"official validity record differs")
    require(official["bootstrap"]["seed"]==protocol["bootstrap_seed"] and official["bootstrap"]["draws"]==protocol["bootstrap_draws"],"official bootstrap settings differ")
    result["official_score_comparison"] = {**comparison,"all_compared_fields_agree":True,
                                            "scores_sha256":digest(official_path)}
    result["independent_recomputation"] = {"classification":classification,"all_primary_gates_pass":all(gates.values()),
        "gates":gates,"contrasts":contrasts,
        "condition_means":{name:{metric:record[metric]["mean"] for metric in METRICS} for name,record in summaries.items()},
        "bootstrap":{"seed":protocol["bootstrap_seed"],"draws":protocol["bootstrap_draws"],"paired":True,"stratified":True}}
    result["all_audit_checks_pass"] = True
    result["protocol_path"] = str(protocol_path.relative_to(repository))
    result["manifest_path"] = str(manifest_path.relative_to(repository))
    result["outcomes_path"] = str(directory.relative_to(repository))
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration",help="Optional expected commit; default is the completed run's recorded registration")
    parser.add_argument("--root",type=Path,default=Path("outputs/experiment6"),help="Result root containing confirmatory/")
    parser.add_argument("--protocol",type=Path,default=Path("experiment6/protocol.json"))
    parser.add_argument("--manifest",type=Path,default=Path("outputs/experiment6/confirmation_manifest.json"))
    parser.add_argument("--directory",type=Path,help="Optional explicit saved-outcome directory")
    parser.add_argument("--output",type=Path,help="Default: ROOT/independent_audit.json; existing records are never overwritten")
    args=parser.parse_args()
    repository=Path(__file__).resolve().parents[1]
    directory=(repository/(args.directory or args.root/"confirmatory")).resolve()
    for required in ("finished.json","scores.json"):
        if not (directory/required).is_file():
            parser.error(f"wait for completed run and scoring: missing {required}")
    registration=args.registration or read_json(directory/"finished.json")["registration"]
    output=(repository/(args.output or args.root/"independent_audit.json")).resolve()
    if output.exists():
        parser.error("audit record already exists; preserve it and choose a new --output path")
    try:
        result=audit(repository,directory,registration,args.protocol,args.manifest)
    except Exception as error:
        result={"audit_kind":"independent_saved_output_audit_after_registration",
                "all_audit_checks_pass":False,"error_type":type(error).__name__,"error":str(error),
                "audit_tool_sha256":digest(__file__),"at_utc":datetime.now(timezone.utc).isoformat()}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        raise
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"audit_pass":True,"classification":result["independent_recomputation"]["classification"],
                      "comparison":result["official_score_comparison"],"output":str(output)},indent=2))


if __name__ == "__main__":
    main()
