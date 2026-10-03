"""Postprocess saved E2 results; never import/evaluate models or alter inputs.

Usage: .venv/bin/python -m experiment2.report
Requires completed confirmatory summary/scores and the public registration lock.
"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


ROOT = Path("outputs/experiment2")
FAMILIES = ("pair", "triple_rescue", "cross_layer")
LABELS = {"ai": "AI forecast", "zero_effect": "Zero effect", "additive": "Additive",
          "multiplicative": "Multiplicative", "count_linear": "Head-count linear",
          "count_logit": "Head-count logit", "case_logit_additive": "Per-case logit additive",
          "case_logit_count": "Per-case logit head-count"}


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def number(value, digits=4):
    return f"{value:.{digits}f}"


def interval(bounds):
    return f"[{bounds[0]:.5f}, {bounds[1]:.5f}]"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def main():
    required = [ROOT / "confirmatory/summary.json", ROOT / "confirmatory/scores.json",
                ROOT / "remote_lock.json", ROOT / "confirmatory/start.json",
                Path("experiment2/predictions.json"), Path("experiment2/conditions.json"),
                Path("experiment2/baseline_predictions.json"), ROOT / "confirmatory/global_decisions.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Reporting requires completed saved outcomes: " + ", ".join(missing))
    summary, scores, lock, start, forecasts, sets, baselines, decisions = map(read, required)
    conditions = sets["confirmatory"]
    by_id = {condition["id"]: condition for condition in conditions}
    novel = [condition for condition in conditions if condition["group"] in FAMILIES]
    seeds = [str(seed) for seed in range(6)]
    if len(novel) != 92 or set(summary) != set(seeds) or set(scores) != set(seeds):
        raise ValueError("Expected 92 novel conditions and all six model seeds")
    if any(summary[s]["n"] != 2048 or scores[s]["n"] != 2048 for s in seeds):
        raise ValueError("Expected the frozen 2048-case confirmation set")
    for s in seeds:
        if set(scores[s]["conditions"]) != set(by_id):
            raise ValueError(f"Seed {s}: score conditions do not match the frozen specification")
        for name in by_id:
            if scores[s]["conditions"][name]["predicted"] != forecasts["expected_probability"][s][name]:
                raise ValueError(f"Seed {s}/{name}: scored prediction differs from frozen prediction")
    if start["preregistration_commit"] != lock["commit"]:
        raise ValueError("Outcome start record and public lock identify different commits")
    methods = list(scores["0"]["scores"])
    baseline_methods = [method for method in methods if method != "ai"]
    if not baseline_methods or any(set(baseline_methods) != set(baselines[s]) for s in seeds):
        raise ValueError("Scored baselines do not match the frozen baseline file")
    gates = {s: scores[s]["prespecified_prediction_gates"] for s in seeds}
    validity = {s: scores[s]["validity_gates"] for s in seeds}
    computed = {
        "all_six_valid": all(all(values.values()) for values in validity.values()),
        "all_six_predictively_adequate": all(values["balanced_rmse_le_0_10"] and
             values["at_least_80pct_novel_within_0_10"] for values in gates.values()),
        "all_six_beat_best_baseline": all(values["beats_best_baseline_ci95"] for values in gates.values())}
    computed["full_claim_pass"] = all(computed.values())
    if computed != decisions:
        raise ValueError("Global decisions disagree with per-seed gates")
    success = decisions["full_claim_pass"]
    verdict = ("The full prespecified all-six-seed claim passed."
               if success else "The full prespecified all-six-seed claim did not pass.")
    stamp = datetime.now(timezone.utc).isoformat()
    commit = lock["commit"]
    lines = ["# Experiment 2: predicting attention-head interventions", "", verdict, "",
             table(["Global decision", "Passed"], [[name, "yes" if passed else "no"] for name, passed in decisions.items()]), "",
             "Predictive adequacy and superiority to baselines are separate claims. Failing superiority alone does not "
             "mean the forecasts missed the registered accuracy tolerances. A failed validity gate prevents the full "
             "scientific claim regardless of numerical forecast performance.", "",
             "The primary endpoint is mean original-target probability under a frozen intervention. "
             "This experiment tests whether discovery-only forecasts predict unseen combinations of head "
             "ablations and rescues. It does not equate an ablation effect with strict necessity during natural computation.", "",
             f"Public preregistration: [{commit}](https://github.com/posix4e/ai-understanding-ai/commit/{commit}). "
             f"Remote contents verified at `{lock['verified_at_utc']}`; confirmation started at `{start['utc']}`. "
             f"This report was generated at `{stamp}`.", "",
             "## Design and scope", "",
             "Six locally trained CPU transformers have 70,720 parameters each, two layers, four heads per layer, "
             "width 64, and MLP width 128. Seeds 0–2 reuse the Experiment 1 models; seeds 3–5 are newly trained "
             "replications. The task presents four random key–value associations and queries one key. "
             "There are 16 possible value classes (uniform chance 6.25%); choosing randomly among the four displayed "
             "values gives 25%. The same frozen 2,048 confirmation cases are evaluated in each model.", "",
             "The intervention site is each head's weighted value output before concatenation and output projection. "
             "Layer 1 interventions act at all four value positions (1, 3, 5, 7); Layer 2 interventions act at query "
             "position 8. Means are position-specific and frozen from 512 independent calibration cases. "
             "Discovery uses 1,024 cases and singleton/all-head conditions. Confirmation contains 145 conditions, "
             "including 92 novel ones: 36 pair ablations, 24 sole-head rescues (equivalent to triple-head corruptions), "
             "and 32 cross-layer compositions. Equivalent rescue/triple conditions are scored once. "
             "Cross-layer interventions retain the naturally recomputed unselected downstream heads.", "",
             "The three novel families receive equal weight in MSE. RMSE is the square root of that balanced MSE. "
             "All six seeds must individually meet all three forecast gates: balanced RMSE ≤0.10; at least 74/92 "
             "novel errors within 0.10; and an upper 95% paired-bootstrap bound strictly below zero for "
             "AI MSE minus the best baseline MSE. Familiar conditions and controls are descriptive, not primary scored evidence.", "",
             "Each seed also requires clean accuracy ≥95%, maximum no-op probability error ≤10⁻⁶, and finite probabilities. "
             "The full claim requires all six seeds to pass validity, predictive adequacy, and baseline superiority.", "",
             "## Every seed and every gate", ""]
    gate_rows = []
    for s in seeds:
        score = scores[s]
        gap = score["ai_minus_best_baseline"]
        gate_rows.append([s, "reused" if int(s) < 3 else "new",
                          number(summary[s]["conditions"]["clean"]["accuracy"]["mean"]),
                          number(score["scores"]["ai"]["balanced_rmse"]),
                          f"{score['novel_within_0_10']}/92 ({score['novel_fraction_within_0_10']:.1%})",
                          number(gap["difference"], 5), interval(gap["bootstrap_ci95"]),
                          ", ".join(name for name, passed in gates[s].items() if not passed) or "none",
                          ", ".join(name for name, passed in validity[s].items() if not passed) or "none",
                          f"{summary[s]['noop_max_probability_error']:.3g}"])
    lines += [table(["Seed", "Cohort", "Clean accuracy", "AI RMSE", "Within 0.10", "AI−best MSE",
                     "Paired 95% interval", "Failed forecast gates", "Failed validity gates", "Max no-op probability error"], gate_rows), "",
              "The no-op error is the maximum per-case target-probability difference across both layers. "
              "A zero value supports implementation identity for these tested inputs; it is not an independent forecast success. "
              "All recorded gate decisions above are taken directly from the frozen scoring procedure.", "",
              f"## Comparison with all {len(baseline_methods)} baselines", ""]
    lines.append(table(["Seed", "Method", "Balanced MSE", "Balanced RMSE", "95% bootstrap MSE interval"],
                       [[s, LABELS.get(method, method), number(scores[s]["scores"][method]["balanced_mse"], 5),
                         number(scores[s]["scores"][method]["balanced_rmse"]),
                         interval(scores[s]["scores"][method]["bootstrap_ci95_mse"])]
                        for s in seeds for method in methods]))
    lines += ["", f"The comparison interval uses AI loss minus the minimum of the {len(baseline_methods)} baseline losses within each "
              "paired resample; the selected baseline may change across resamples. Point-estimate best baselines are " +
              "; ".join(f"seed {s}: {LABELS.get(scores[s]['ai_minus_best_baseline']['best_point_baseline'], scores[s]['ai_minus_best_baseline']['best_point_baseline'])}" for s in seeds) + ".", "",
              "The per-case logit additive and per-case logit head-count baselines use paired raw discovery probabilities, "
              "matching the AI forecast's access to case-level discovery information. They were fixed before confirmation.", ""]
    cohort_rows = []
    for label, members in [("Reused seeds 0–2", seeds[:3]), ("New seeds 3–5", seeds[3:]), ("All six", seeds)]:
        for method in methods:
            mse = mean(scores[s]["scores"][method]["balanced_mse"] for s in members)
            cohort_rows.append([label, LABELS.get(method, method), number(mse, 5), number(math.sqrt(mse))])
    lines += ["### Reused versus newly trained seeds", "",
              table(["Cohort", "Method", "Mean seed MSE", "Square root of mean seed MSE"], cohort_rows), "",
              "Cohort summaries are descriptive equal-seed averages, not additional pooled hypothesis tests. "
              "A favorable cohort average does not override a failed seed or satisfy the all-six primary rule.", "",
              "## Where the forecasts failed or succeeded", ""]
    family_rows = []
    for s in seeds:
        for family in FAMILIES:
            ai_mse = scores[s]["group_scores"]["ai"][family]
            best = min(baseline_methods, key=lambda m: scores[s]["group_scores"][m][family])
            best_mse = scores[s]["group_scores"][best][family]
            family_rows.append([s, family, number(math.sqrt(ai_mse)), LABELS.get(best, best),
                                number(math.sqrt(best_mse)), "yes" if math.sqrt(ai_mse) > .10 else "no"])
    lines += [table(["Seed", "Novel family", "AI RMSE", "Best point baseline", "Baseline RMSE", "Family RMSE >0.10"], family_rows), "",
              "Family rows reveal localized failures; the 0.10 family flag is descriptive rather than an additional "
              "registered gate. No family is dropped from the primary score.", ""]
    worst = sorted([(abs(scores[s]["conditions"][c["id"]]["error"]), s, c["id"])
                    for s in seeds for c in novel], reverse=True)
    worst_per_seed = [next(row for row in worst if row[1] == s) for s in seeds]
    def error_rows(rows):
        return [[s, name, number(scores[s]["conditions"][name]["predicted"]),
                 number(scores[s]["conditions"][name]["observed"]),
                 interval(scores[s]["conditions"][name]["ci95"]), number(error)]
                for error, s, name in rows]
    lines += ["Largest novel error in each seed:", "",
              table(["Seed", "Condition", "Forecast", "Observed", "Observed mean 95% interval", "Absolute error"], error_rows(worst_per_seed)),
              "", "Largest 12 errors across all novel seed–condition combinations:", "",
              table(["Seed", "Condition", "Forecast", "Observed", "Observed mean 95% interval", "Absolute error"], error_rows(worst[:12])), "",
              "## Descriptive head effects", "",
              "Each cell reports target probability / task accuracy on confirmation cases. Head indices are 0–3. "
              "These singleton/all-head conditions were available during discovery and are not novel successes.", ""]
    def pair(s, name):
        metrics = summary[s]["conditions"][name]
        return f"{metrics['target_probability']['mean']:.4f} / {metrics['accuracy']['mean']:.4f}"
    lines.append(table(["Seed", "Layer", "Corruption", "Head 0", "Head 1", "Head 2", "Head 3", "All heads"],
                       [[s, layer, kind, *[pair(s, f"L{layer}_{kind}_h{head}") for head in range(4)],
                         pair(s, f"L{layer}_{kind}_all")]
                        for s in seeds for layer in (1, 2) for kind in ("mean", "zero", "resample")]))
    lines += ["", "## Matched donors and other controls", "",
              "Resampled donors are valid dictionaries matched on query key and query-pair position. Active donors "
              "change the queried value; matched-control donors preserve the queried association. An active-versus-matched "
              "difference therefore bears on association-specific information, while the other dictionary entries can still "
              "differ. Resampling is not interchangeable with zero/mean ablation, and a weak active-donor effect does not "
              "prove that the patched heads carry no useful information.", "",
              table(["Seed", "Layer", "Active donor, all heads P / accuracy", "Matched donor, all heads P / accuracy",
                     "Matched singleton probability range"],
                    [[s, layer, pair(s, f"L{layer}_resample_all"), pair(s, f"L{layer}_matched_all"),
                      f"{min(summary[s]['conditions'][f'L{layer}_matched_h{h}']['target_probability']['mean'] for h in range(4)):.4f}–"
                      f"{max(summary[s]['conditions'][f'L{layer}_matched_h{h}']['target_probability']['mean'] for h in range(4)):.4f}"]
                     for s in seeds for layer in (1, 2)]), "",
              table(["Seed", "Layer", "Wrong-position mean P / accuracy", "Wrong-position zero", "Wrong-position resample", "MLP zero", "MLP mean"],
                    [[s, layer, *[pair(s, f"L{layer}_{kind}_wrong_position") for kind in ("mean", "zero", "resample")],
                      pair(s, f"L{layer}_mlp_zero"), pair(s, f"L{layer}_mlp_mean")]
                     for s in seeds for layer in (1, 2)]), "",
              "Wrong-position controls patch Layer 1 at the query instead of values and Layer 2 at values instead of the query. "
              "MLP controls alter a different component at the intended positions; an MLP effect is possible and is not "
              "automatically a failed negative control. Full per-condition forecasts, observations, accuracy, and errors are "
              "in [forecasts_vs_results.csv](forecasts_vs_results.csv).", "",
              "## Uncertainty and interpretation", "",
              "The AI and case-logit-additive forecasts are identical on all 32 cross-layer conditions by construction. "
              "Those conditions test their shared cross-layer assumption, not incremental value from the AI explanation. "
              "Any advantage over that baseline must come from the within-layer interaction allocation.", "",
              "Primary loss and loss-difference intervals use 2,000 paired case bootstraps with fixed seed 6200001, "
              "preserving correlation across conditions and using shared case weights across model seeds. They are "
              "conditional on the six trained models, selected task, calibration means, discovery-derived forecasts, "
              "and intervention definitions. They do not incorporate a population distribution over training seeds, "
              "uncertainty from model selection, or multiple-task generalization. Displayed observed-condition mean "
              "intervals are normal standard-error intervals, not bootstrap intervals; accuracy intervals in the source "
              "summary are Wilson intervals. The ±0.10 forecast tolerance is a practical error criterion, not a claimed "
              "coverage probability.", "",
              "Experiment 1's association-lookup account need not predict how information is distributed redundantly "
              "across attention heads. Experiment 2 tests that stronger quantitative prediction. A failure here narrows "
              "the supported claim even when the task algorithm is correctly identified. Conversely, passing these tests "
              "would support this frozen forecast rule on this local task, not a general claim that AI understands arbitrary AI systems. "
              "Mean/zero ablations change internal states, and clean-head rescue is a controlled sufficiency test within "
              "an altered network. Neither alone establishes strict necessity or sufficiency in ordinary unmodified execution.", "",
              "All experiment computation used local CPU code; no model inference API was used. Experiment 2 code and data "
              "are namespaced under `experiment2/` and `outputs/experiment2/`; frozen Experiment 1 files were not edited.", "",
              "## Reproduction", "",
              "Inspect the public registration commit and its manifest before executing. The registered phase controller "
              "refuses to overwrite an existing confirmation directory and verifies frozen inputs against the public lock. "
              "Archive the existing output directory before a deliberate rerun; use a fresh local working copy or preserve "
              "the original tracked outputs. Do not rerun discovery or regenerate calibration means as part of reproducing confirmation.", "",
              "```sh", "# Pure reporting from existing saved results:", ".venv/bin/python -m experiment2.report", "",
              "# Deliberate confirmatory rerun; preserve the originals first:",
              "cp -R outputs/experiment2/confirmatory outputs/experiment2/confirmatory_original",
              "mv outputs/experiment2/confirmatory outputs/experiment2/confirmatory_preserved",
              ".venv/bin/python -m experiment2.run_phase --phase confirmatory",
              ".venv/bin/python -m experiment2.score", ".venv/bin/python -m experiment2.report", "```", "",
              "The archive destination names above must not already exist. Reruns create new runtime timestamps; they "
              "do not create a new preregistration or replace the original evidentiary run.", ""]
    csv_path = ROOT / "forecasts_vs_results.csv"
    fields = ["seed", "cohort", "condition", "group", "novel", "predicted_probability", "observed_probability",
              "observed_ci95_low", "observed_ci95_high", "signed_error", "absolute_error", "accuracy",
              "within_0_10", *[f"baseline_{method}" for method in baseline_methods], "operations_json"]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for s in seeds:
            for condition in conditions:
                name = condition["id"]
                row = scores[s]["conditions"][name]
                writer.writerow({"seed": s, "cohort": "reused" if int(s) < 3 else "new",
                                 "condition": name, "group": condition["group"], "novel": condition["group"] in FAMILIES,
                                 "predicted_probability": row["predicted"], "observed_probability": row["observed"],
                                 "observed_ci95_low": row["ci95"][0], "observed_ci95_high": row["ci95"][1],
                                 "signed_error": row["error"], "absolute_error": abs(row["error"]),
                                 "accuracy": row["accuracy"], "within_0_10": abs(row["error"]) <= .10,
                                 **{f"baseline_{method}": baselines[s][method][name] for method in baseline_methods},
                                 "operations_json": json.dumps(condition["ops"], separators=(",", ":"))})
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(ROOT / "RESULTS.md"), "csv": str(csv_path),
                      "primary_all_six_passed": success, "seed_condition_rows": len(seeds) * len(conditions)}))


if __name__ == "__main__":
    main()
