"""Pure E4 reporting from frozen postprocessed outcomes; no model execution."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path("outputs/experiment4")


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def value_ci(result):
    bounds = result.get("ci95_bootstrap", result.get("ci95_wilson"))
    return f"{result['mean']:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"


def main():
    paths = [ROOT / "confirmatory/scores.json", ROOT / "confirmatory/global_decisions.json",
             ROOT / "confirmatory/forecast_statistics.json", ROOT / "remote_lock.json",
             ROOT / "confirmatory/start.json", Path("experiment4/protocol.json"), Path("experiment4/conditions.json")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Completed E4 scores required: " + ", ".join(missing))
    scores, decisions, numeric, lock, start, protocol, conditions = map(read, paths)
    seeds = [str(seed) for seed in protocol["model_seeds"]]
    if set(scores) != set(seeds) or any(scores[s]["n"] != protocol["n"] for s in seeds):
        raise ValueError("Incomplete registered model/case grid")
    expected = {"all_six_valid": all(all(scores[s]["validity_gates"].values()) for s in seeds),
                "all_six_recovery": all(all(scores[s]["recovery_gates"].values()) for s in seeds),
                "all_six_specificity": all(all(scores[s]["specificity_gates"].values()) for s in seeds)}
    expected["full_claim_pass"] = all(expected.values())
    if expected != decisions:
        raise ValueError("Global decisions disagree with individual gates")
    commit = lock["commit"]
    if start["preregistration_commit"] != commit:
        raise ValueError("Runtime registration differs from public lock")
    correct = protocol["correct_guard"]
    verdict = ("The full prespecified six-model claim passed." if decisions["full_claim_pass"]
               else "The full prespecified six-model claim did not pass.")
    recovery = ("The correct guard met every registered recovery gate in all six models."
                if decisions["all_six_recovery"] else
                "The correct guard did not meet every registered recovery gate across all six models.")
    specificity = ("The registered advantage over the mean of all eight matched controls passed in all six models."
                   if decisions["all_six_specificity"] else
                   "The registered advantage over the mean of all eight matched controls did not pass in every model.")
    lines = ["# Experiment 4: restoring causal key visibility", "", verdict, "", recovery + " " + specificity, "",
             "Validity, recovery and specificity are separate decisions. A positive recovery result does not override "
             "a specificity failure, and algebraically guaranteed oracle checks do not count as empirical discoveries.", "",
             table(["Global decision", "Passed"], [[name, "yes" if passed else "no"] for name, passed in decisions.items()]), ""]
    seed_rows = []
    for s in seeds:
        row = scores[s]
        guard = row["conditions"][correct]
        minimum_query = min(guard["per_query"][str(q)]["accuracy"]["mean"] for q in range(4))
        failed = [name for group in ("validity_gates", "recovery_gates", "specificity_gates")
                  for name, passed in row[group].items() if not passed]
        seed_rows.append([s, f"{row['conditions']['native']['accuracy']['mean']:.4f}",
                          f"{row['conditions']['grouped_canonical']['accuracy']['mean']:.4f}",
                          f"{guard['accuracy']['mean']:.4f}", f"{minimum_query:.4f}",
                          value_ci(row["comparisons"]["guard_minus_grouped"]),
                          value_ci(row["comparisons"]["guard_minus_mean_eight"]), ", ".join(failed) or "none"])
    lines += ["## Six-seed result", "",
              table(["Seed", "Native accuracy", "Grouped accuracy", "Guard accuracy", "Minimum query accuracy",
                     "Guard−grouped P [95% CI]", "Guard−mean 8 P [95% CI]", "Failed gates"], seed_rows), "",
              "Recovery requires accuracy ≥0.95 in every query stratum and a paired probability-improvement lower "
              "95% bound strictly above 0.30. Specificity requires the lower bound versus the equally weighted eight-control "
              "mean strictly above 0.05. The full claim requires these plus every validity check in every seed.", "",
              "## Prospective record and scope", "",
              f"Public preregistration: [{commit}](https://github.com/posix4e/ai-understanding-ai/commit/{commit}). "
              f"Remote contents verified at `{lock['verified_at_utc']}`; confirmation started at `{start['utc']}`. "
              f"Report generated at `{datetime.now(timezone.utc).isoformat()}`.", "",
              "Experiment 3's failed strong position-label claim is preserved. Its published outcomes motivated this "
              "new prospective follow-up; no E4 trained-model discovery or new fitting occurred. Six existing local CPU "
              "checkpoints (70,720 parameters, two layers, four heads each) evaluated the same 2,048 fresh dictionaries, "
              "balanced at 512 queries per pair index. The data audit excludes prior association dictionaries across all "
              "24 pair orders and four query choices, and excludes repeated unordered dictionaries within E4. "
              "All 15 conditions and all failures are retained. No inference API or new training was used.", "",
              "## Intervention and controls", "",
              "Grouping all four keys before all four values exposes each first-layer value node to keys it could not "
              "see in the training order. Canonical position IDs remain attached to their original tokens. The correct "
              "guard removes Vi→Kj edges for j>i, using a stable masked softmax from the grouped model's own cached Q/K. "
              "It edits every head's value rows only, uses no native activations or target-dependent decisions, and "
              "retains each value's own key.", "",
              "Nine masks exhaust the row-degree-matched family: V0 keeps only K0; V1 keeps K1 plus A∈{0,2,3}; "
              "V2 keeps all keys except B∈{0,1,3}; V3 keeps all keys. Correct is A=0,B=3. Every mask removes six key "
              "edges per head with the same per-row counts (3,2,1,0). The masks are not matched on removed attention "
              "mass or activation-change norm. V0 has the same mask in every condition, so this comparison cannot "
              "isolate its edge identities.", "",
              "The guard restores first-layer value states by algebra, and the final-query first-layer state is "
              "permutation-invariant. Grouped key states remain altered because they lost access to earlier values. "
              "Those changed key states can affect layer 2, so final behavioral recovery by the guard alone is empirical. "
              "Native key/value-state transplants are explicitly labeled oracle controls. The complete guard-plus-key "
              "oracle must reproduce native output; the value-only oracle must reproduce the guard. Neither identity "
              "is evidence of a new learned repair mechanism.", "",
              "## Every matched guard", "",
              "Accuracy intervals below are individual Wilson 95% intervals, not simultaneous intervals. Probability "
              "intervals use shared paired bootstrap rows. Query accuracies are ordered q=0,1,2,3.", ""]
    guard_conditions = [c for c in conditions if c["family"] == "matched_guard"]
    lines.append(table(["Seed", "Guard", "Original probability [95% CI]", "Accuracy [95% Wilson CI]", "Four query accuracies"],
                       [[s, c["id"], value_ci(scores[s]["conditions"][c["id"]]["original_probability"]),
                         value_ci(scores[s]["conditions"][c["id"]]["accuracy"]),
                         ", ".join(f"{scores[s]['conditions'][c['id']]['per_query'][str(q)]['accuracy']['mean']:.4f}" for q in range(4))]
                        for s in seeds for c in guard_conditions]))
    lines += ["", "Every condition, including references and oracles, is in [condition_results.csv](condition_results.csv). "
              "Complete query-stratified means, probability intervals and Wilson intervals are in "
              "[per_query_results.csv](per_query_results.csv). No low-performing guard or query stratum was dropped.", "",
              "### Mean and best matched controls", "",
              table(["Seed", "Mean of 8 wrong masks P [95% CI]", "Best point mask", "Best point P", "Guard−best P [reselected 95% CI]"],
                    [[s, value_ci(scores[s]["comparisons"]["mean_eight_wrong_probability"]),
                      scores[s]["comparisons"]["guard_minus_best_wrong"]["best_point_condition"],
                      f"{scores[s]['comparisons']['guard_minus_best_wrong']['best_point_probability']:.4f}",
                      value_ci(scores[s]["comparisons"]["guard_minus_best_wrong"])] for s in seeds]), "",
              "The best-control interval reselects the highest-mean wrong mask within each shared bootstrap draw. "
              "It is descriptive and is not an extra success gate. Passing the mean-control comparison does not "
              "establish that the correct guard beats every individual mask or is uniquely effective. All eight paired "
              "guard-minus-control effects and intervals are in [control_comparisons.csv](control_comparisons.csv).", "",
              "## Validity and oracle checks", "",
              table(["Seed", "Native accuracy", "Grouped no-op max P error", "Guard L1 value max error", "Guard L1 query max error",
                     "Guard key change vs grouped", "Guard−value oracle max class-P error", "Full oracle−native max class-P error"],
                    [[s, f"{scores[s]['conditions']['native']['accuracy']['mean']:.4f}",
                      *[f"{scores[s]['manipulation_errors'][name]:.3g}" for name in
                        ("grouped_noop_probability", "correct_guard_l1_value_native", "correct_guard_l1_query_native",
                         "correct_guard_l1_key_grouped", "guard_vs_value_oracle_full_probability", "full_oracle_vs_native_full_probability")]] for s in seeds]), "",
              "Native accuracy must reach 0.95. Grouped no-op error must be ≤10⁻⁶; the five residual/output identities "
              "must be ≤10⁻⁵. Output-oracle errors cover every case and all 16 classes, rather than only the correct "
              "class. Finite complete arrays and full-probability/argmax/accuracy consistency are also checked. "
              "Actual first-layer output diagnostics and effective layer-2 input diagnostics are stored separately: "
              "transplanting a state at layer-2 input does not retroactively change the first-layer output.", "",
              "## Keys and guard factorial (descriptive)", "",
              table(["Seed", "Keys effect without guard [95% CI]", "Keys effect with guard [95% CI]",
                     "Interaction: full−guard−keys+grouped [95% CI]"],
                    [[s, value_ci(scores[s]["factorial_descriptive"]["keys_effect_without_guard"]),
                      value_ci(scores[s]["factorial_descriptive"]["keys_effect_with_guard"]),
                      value_ci(scores[s]["factorial_descriptive"]["interaction_full_minus_guard_minus_keys_plus_grouped"])] for s in seeds]), "",
              "These paired probability-scale effects compare grouped, guard, native-keys restoration, and combined "
              "guard-plus-native-keys restoration. They help describe remaining dependencies and carry no new gate. "
              "The combined condition is an oracle with an algebraic output-restoration guarantee.", "",
              "## All 180 prospective numeric forecasts", "",
              "The frozen predictor assigned original-target probability and accuracy to 15 conditions in each of "
              "six models. Every error is evaluated against absolute tolerance 0.15, separately from causal gates. "
              "Reference and oracle forecasts are explicitly separated so known identities do not inflate claims "
              "about predicting novel intervention effects.", "",
              table(["Scope", "Forecast count", "Within ±0.15", "Fraction", "MAE", "RMSE", "Maximum error"],
                    [[scope, result["count"], result["within_tolerance"], f"{result['fraction_within_tolerance']:.1%}",
                      f"{result['mae']:.4f}", f"{result['rmse']:.4f}", f"{result['max_absolute_error']:.4f}"] for scope, result in numeric.items()]), "",
              table(["Seed", "All forecasts within ±0.15", "Guard forecasts within ±0.15", "All RMSE", "Guard RMSE"],
                    [[s, f"{scores[s]['forecast_statistics']['all_15']['within_tolerance']}/30",
                      f"{scores[s]['forecast_statistics']['matched_guards_9']['within_tolerance']}/18",
                      f"{scores[s]['forecast_statistics']['all_15']['rmse']:.4f}",
                      f"{scores[s]['forecast_statistics']['matched_guards_9']['rmse']:.4f}"] for s in seeds]), "",
              "Every forecast is in [forecasts_vs_results.csv](forecasts_vs_results.csv); every miss is listed in "
              "[forecast_misses.csv](forecast_misses.csv). This error tolerance is a practical criterion, not a "
              "forecast-coverage probability.", "",
              "## Attention and uncertainty", "",
              "All four attention heads remain visible without selecting favorable ones. Per-condition head means "
              "for first-layer value→true-key attention and second-layer query→correct-value attention are in "
              "[attention_diagnostics.csv](attention_diagnostics.csv). They are descriptive measurements, not "
              "additional causal tests. Residual diagnostics, every query contrast, and all gate decisions remain "
              "in [scores.json](confirmatory/scores.json).", "",
              "Mean intervals use 2,000 dictionary-row multinomial bootstraps, stratified within the four query "
              "indices with fixed seed 8200001. The same weights are shared across every condition and model. "
              "The eight wrong controls are averaged within each row before resampling. Uncertainty is conditional "
              "on these six checkpoints and input distribution, not a population distribution over trained models. "
              "Individual Wilson and condition intervals are not simultaneous intervals.", "",
              "## Interpretation and preservation", "",
              "The stated claim concerns empirical sufficiency of a specific causal-visibility repair and its "
              "advantage over the mean of the complete row-degree-matched control family. It does not establish "
              "necessity of every removed edge, a unique complete algorithm, spontaneous grouped-layout transfer, "
              "or a universal transformer mechanism. The guard supplies an externally chosen training-prefix "
              "mask, so it is not a learned adaptation by the model.", "",
              "This is a prospective extension of positional-binding work, with no field-wide novelty claim; see "
              "[the earlier primary-source audit](../experiment3/RELATED_WORK.md). All E1–E3 frozen results, including "
              "E3's negative primary result, remain unchanged. Positive subresults here do not relabel earlier failures. "
              "Any subsequent refinement requires its own prospective data and registration.", "",
              "## Reproduction", "",
              "The registration fixes inputs, checkpoint hashes, intervention code, forecasts and score rules. "
              "These commands only recompute scores and reporting from saved outcomes; preserve existing generated "
              "files before a deliberate rerun if their original runtime record is needed. The trained-model phase "
              "controller separately verifies the public lock and refuses an existing confirmation directory.", "",
              "```sh", ".venv/bin/python -m experiment4.score", ".venv/bin/python -m experiment4.report", "```", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    comparison_rows = []
    for s in seeds:
        for name, result in scores[s]["comparisons"]["guard_minus_each_wrong"].items():
            comparison_rows.append({"seed": s, "wrong_guard": name, "paired_probability_difference": result["mean"],
                                    "ci95_low": result["ci95_bootstrap"][0], "ci95_high": result["ci95_bootstrap"][1]})
    with (ROOT / "control_comparisons.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader(); writer.writerows(comparison_rows)
    print(json.dumps({"report": str(ROOT / "RESULTS.md"), "decisions": decisions,
                      "numeric_forecasts": numeric["all_15"]["count"],
                      "numeric_misses": numeric["all_15"]["count"] - numeric["all_15"]["within_tolerance"]}))


if __name__ == "__main__":
    main()
