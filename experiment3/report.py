"""Generate E3 reporting from completed scores only; never execute a model."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path("outputs/experiment3")


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
             ROOT / "confirmatory/start.json", Path("experiment3/protocol.json"), Path("experiment3/conditions.json")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Completed E3 outcomes required: " + ", ".join(missing))
    scores, decisions, numeric, lock, start, protocol, conditions = map(read, paths)
    seeds = [str(seed) for seed in protocol["model_seeds"]]
    if set(scores) != set(seeds) or any(scores[s]["n"] != protocol["n"] for s in seeds):
        raise ValueError("Incomplete registered model/case grid")
    expected = {"all_six_valid": all(all(scores[s]["validity_gates"].values()) for s in seeds),
                "all_six_mechanism_gates": all(all(scores[s]["mechanism_gates"].values()) for s in seeds)}
    expected["full_claim_pass"] = all(expected.values())
    if expected != decisions:
        raise ValueError("Global decisions do not match individual seed gates")
    commit = lock["commit"]
    if start.get("preregistration_commit", start.get("commit")) != commit:
        raise ValueError("Runtime start record differs from registration lock")
    verdict = ("The full prespecified six-model mechanistic claim passed."
               if decisions["full_claim_pass"] else "The full prespecified six-model mechanistic claim did not pass.")
    lines = ["# Experiment 3: positional coordinates and key–value binding", "", verdict, "",
             table(["Global decision", "Passed"], [[name, "yes" if passed else "no"] for name, passed in decisions.items()]), "",
             "Numeric forecast accuracy is a separate result from the registered mechanistic gates; all forecast misses "
             "and every failed condition are retained below and in the accompanying files.", "",
             "This study is a prospective test and systematic extension of prior positional-steering work. "
             "It makes no priority claim for discovering positional key–value binding; the distinctive evidence sought "
             "here is the frozen permutation grid, specifically predicted wrong answers, matched coherent controls, "
             "and replication across six existing models.", "",
             "The [primary-source related-work audit](RELATED_WORK.md) identifies close precedents and "
             "explains the narrower scope of this test.", "",
             f"Preregistration: [{commit}](https://github.com/posix4e/ai-understanding-ai/commit/{commit}). "
             f"Public contents verified at `{lock['verified_at_utc']}`; confirmation started at `{start['utc']}`. "
             f"Report generated at `{datetime.now(timezone.utc).isoformat()}`.", "",
             "## What was tested", "",
             "The same six frozen, locally trained CPU transformers from Experiments 1–2 were reused; each has "
             "70,720 parameters, two layers and four attention heads per layer. No E3 training, shifted-layout discovery, "
             "or per-model fitting occurred. Each of 2,048 fresh dictionaries contributes one query, with 512 cases "
             "at each of four query-pair indices. All 50 conditions use the same rows in every model.", "",
             "The original sequence alternates four keys and values before a final query. The grouped layout presents "
             "all four keys, then all four values, then the query. Canonical positional coordinates travel with their "
             "original tokens. For a permutation π, a coherent map assigns key/value pair i coordinates 2π(i) and "
             "2π(i)+1; a value-only map leaves key i at 2i but assigns its value coordinate 2π(i)+1. The binding account "
             "predicts the value at inverse-π(query-pair) for value-only maps, and the original answer for coherent maps. "
             "All keys precede all values, and query token/position remain unchanged. Input residual patching changes "
             "the supplied token-plus-position embedding sum; model weights remain frozen.", "",
             "All 24 permutations were fixed independently of outcomes. Collapsing identity maps leaves 23 coherent "
             "and 23 value-only maps plus four references. The primary test uses all nine value derangements and their "
             "nine matched coherent controls. Each value condition must reach slot accuracy ≥0.90, each coherent "
             "control original accuracy ≥0.95, and each seed's equally weighted nine-condition mean of slot minus "
             "original probability must have a paired 95% bootstrap lower bound strictly above 0.80. Validity also "
             "requires native original accuracy ≥0.95 and maximum no-op original-probability error ≤10⁻⁶. Every "
             "condition and all six seeds must pass; averages cannot override individual failures.", "",
             "## Seed-by-seed gates", ""]
    seed_rows = []
    for s in seeds:
        row = scores[s]
        values = [v["mean"] for v in row["primary_conditions"].values() if v["endpoint"] == "slot_accuracy"]
        controls = [v["mean"] for v in row["primary_conditions"].values() if v["endpoint"] == "original_accuracy"]
        failed = [name for name, passed in {**row["validity_gates"], **row["mechanism_gates"]}.items() if not passed]
        seed_rows.append([s, "E1" if int(s) < 3 else "E2 replication",
                          f"{row['conditions']['original_native']['metrics']['original_accuracy']['mean']:.4f}",
                          f"{min(values):.4f}", f"{min(controls):.4f}", value_ci(row["primary_probability_margin"]),
                          f"{row['noop_max_probability_error']:.3g}", ", ".join(failed) or "none"])
    lines += [table(["Seed", "Checkpoint cohort", "Native accuracy", "Minimum of 9 slot accuracies",
                     "Minimum of 9 coherent accuracies", "Slot−original P [95% CI]", "No-op max error", "Failed gates"], seed_rows), "",
              "### Every primary condition", "",
              "Accuracy intervals are individual Wilson 95% intervals. They are not simultaneous intervals across "
              "conditions or models; the registered accuracy gates use the point estimates.", ""]
    primary_rows = []
    for s in seeds:
        for name, result in scores[s]["primary_conditions"].items():
            primary_rows.append([s, name, result["endpoint"], value_ci(result), result["threshold"], "pass" if result["pass"] else "FAIL"])
    lines += [table(["Seed", "Condition", "Endpoint", "Accuracy [individual 95% CI]", "Minimum", "Decision"], primary_rows), "",
              "### Query-stratified probability margin", "",
              table(["Seed", "Query-pair index", "Mean of 9 slot−original probabilities [paired 95% CI]"],
                    [[s, q, value_ci(scores[s]["primary_probability_margin"]["per_query"][str(q)])] for s in seeds for q in range(4)]), "",
              "Complete per-query probability intervals and accuracy intervals for every condition are included in "
              "[scores.json](confirmatory/scores.json). Query subgroup rows are descriptive; they do not add new success gates.", "",
              "## Native, canonical and no-op references", "",
              table(["Seed", "Reference", "Original probability [95% CI]", "Original accuracy [95% CI]"],
                    [[s, name, value_ci(scores[s]["conditions"][name]["metrics"]["original_probability"]),
                      value_ci(scores[s]["conditions"][name]["metrics"]["original_accuracy"])]
                     for s in seeds for name in ("original_native", "original_noop", "grouped_native", "grouped_canonical")]), "",
              "Grouped-native has no parity-based slot map. Its slot fields duplicate the original reference by convention "
              "and are marked `slot_mapping_defined=false`; they are not evidence for the coordinate rule.", "",
              "## All prospective numeric forecasts", "",
              "Four means were forecast for each condition and each model: original probability, slot probability, "
              "original accuracy, and slot accuracy. The full grid therefore contains 1,200 forecasts. Each uses the "
              "fixed absolute-error tolerance 0.15. This is a practical tolerance, not a probabilistic coverage interval. "
              "Primary-only summaries distinguish all 18 primary maps from the nine value derangements alone.", "",
              table(["Scope", "Forecasts", "Within ±0.15", "Fraction", "MAE", "RMSE", "Maximum error"],
                    [[scope, result["count"], result["within_tolerance"], f"{result['fraction_within_tolerance']:.1%}",
                      f"{result['mae']:.4f}", f"{result['rmse']:.4f}", f"{result['max_absolute_error']:.4f}"]
                     for scope, result in numeric.items()]), "",
              table(["Seed", "Scope", "Within ±0.15", "RMSE", "Maximum error"],
                    [[s, scope, f"{result['within_tolerance']}/{result['count']}", f"{result['rmse']:.4f}", f"{result['max_absolute_error']:.4f}"]
                     for s in seeds for scope, result in scores[s]["forecast_statistics"].items()]), "",
              "Every forecast is in [forecasts_vs_results.csv](forecasts_vs_results.csv). Every miss, without filtering "
              "by condition family or seed, is in [forecast_misses.csv](forecast_misses.csv). Native references and "
              "no-op checks are not novel mechanistic successes.", "",
              "## Baseline fidelity to observed answers", "",
              "Fidelity asks whether a rule predicts the model's observed argmax, which differs from task correctness. "
              "The semantic rule always chooses the original associated value; the slot rule chooses the coordinate-implied "
              "value. A uniform displayed-value rule assigns probability 1/4 when the observed argmax is among the four "
              "displayed values, otherwise zero. Uniform over all 16 classes assigns 1/16. The slot rule is the tested "
              "mechanistic prediction, not an independent competitor. Native-grouped slot fidelity is undefined as a "
              "mechanistic rule despite its reference fields.", ""]
    primary_names = [c["id"] for c in conditions if c["primary_group"] == "value_derangement"]
    baseline_rows = []
    for s in seeds:
        for method in ("semantic_original", "uniform_displayed_four", "uniform_sixteen", "slot_rule"):
            fidelity = sum(scores[s]["baseline_fidelity"][name][method]["mean"] for name in primary_names) / 9
            baseline_rows.append([s, method, f"{fidelity:.4f}"])
    lines += [table(["Seed", "Rule", "Mean fidelity across 9 value derangements"], baseline_rows), "",
              "All condition-level fidelity intervals, baseline-assigned original/slot probabilities and discrepancies "
              "from observed model probabilities are in [baseline_fidelity.csv](baseline_fidelity.csv). The displayed "
              "nine-condition averages above are descriptive; do not average their condition-level interval endpoints "
              "to infer a pooled interval.", "",
              "## Secondary rule and matched-control diagnostics", "",
              "Forward-π and inverse-π coincide on involutions and on some fixed queries. The following comparison uses "
              "only the 14 non-involutions and only cases for which the two predicted values differ. Eligible outcomes "
              "are averaged within each dictionary before resampling, preserving row clustering. These are descriptive "
              "secondary diagnostics with no additional gates.", "",
              table(["Seed", "Inverse answer fidelity [95% CI]", "Forward answer fidelity [95% CI]",
                     "Inverse−forward fidelity [95% CI]", "Inverse−forward target P [95% CI]"],
                    [[s, value_ci(scores[s]["secondary_forward_inverse"]["inverse_fidelity"]),
                      value_ci(scores[s]["secondary_forward_inverse"]["forward_fidelity"]),
                      value_ci(scores[s]["secondary_forward_inverse"]["inverse_minus_forward"]),
                      value_ci(scores[s]["secondary_forward_inverse"]["inverse_minus_forward_probability"])] for s in seeds]), "",
              table(["Seed", "Canonical−native grouped original P [95% CI]", "Value−coherent P on the SAME wrong target [95% CI]",
                     "Other two wrong displayed targets, mean P [95% CI]"],
                    [[s, value_ci(scores[s]["secondary_probability_contrasts"]["canonical_minus_native_grouped_original_probability"]),
                      value_ci(scores[s]["matched_coherent_wrong_target"]["paired_difference"]),
                      value_ci(scores[s]["matched_coherent_wrong_target"]["other_two_wrong_displayed_targets_mean_probability"])] for s in seeds]), "",
              "The matched wrong-target contrast uses the same inverse-π target in each value/coherent derangement pair. "
              "It distinguishes specific redirection from an indiscriminate increase in wrong answers. All 23 per-permutation "
              "coherent-minus-value original-probability contrasts and intervals are retained in scores.json, as are the "
              "individual non-involution comparisons and query breakdowns.", "",
              "## Descriptive attention diagnostics", "",
              "All four heads are retained and equally weighted; none was selected using outcomes. Layer 1 statistics "
              "average each head's attention over all four value positions to either the true key or the coordinate-associated "
              "key. Layer 2 statistics use query attention to the original or coordinate-implied value. Each table cell is "
              "the four head means (heads 0,1,2,3), averaged over the nine maps in that family. These summaries do not "
              "establish attention causality or serve as extra success criteria.", ""]
    attention_rows = []
    diagnostics = ("l1_true_key_attention", "l1_slot_key_attention", "l2_original_value_attention", "l2_slot_value_attention")
    for s in seeds:
        for group in ("value_derangement", "coherent_control"):
            names = [c["id"] for c in conditions if c["primary_group"] == group]
            cells = []
            for diagnostic in diagnostics:
                heads = [sum(scores[s]["conditions"][name]["attention"][diagnostic]["heads"][h] for name in names) / len(names) for h in range(4)]
                cells.append(", ".join(f"{x:.3f}" for x in heads))
            attention_rows.append([s, group, *cells])
    lines += [table(["Seed", "Family", "L1 true-key heads", "L1 slot-key heads", "L2 original-value heads", "L2 slot-value heads"], attention_rows), "",
              "All individual condition/head means are in [attention_diagnostics.csv](attention_diagnostics.csv); per-query "
              "head means remain in scores.json. Full class probabilities and per-case attention summaries remain in the raw NPZ files.", "",
              "## Limits and uncertainty", "",
              "All mean intervals use 2,000 query-stratified dictionary-row bootstraps with fixed seed 7200001. The same "
              "weights are shared across conditions and six model seeds. Primary margin aggregation averages the nine "
              "derangements within each row before resampling. These intervals are conditional on these six frozen "
              "models, task distribution and coordinate interventions; they do not quantify variation over a population "
              "of training seeds or tasks. Individual condition intervals are not simultaneous.", "",
              "External positional reassignment is an intervention, not spontaneous layout generalization. The grouped "
              "layout changes causal predecessor sets as well as adjacency relative to interleaved training; canonical "
              "coordinates do not restore the original intermediate attention mask. A successful specific redirection "
              "supports causal control of binding by trained positional coordinates in this setting. It does not establish "
              "a unique complete algorithm, strict necessity of one head, arbitrary-layout competence, or novelty over "
              "all prior literature. Failure of any registered gate must remain visible even if other diagnostics look favorable.", "",
              "All computation was local CPU execution; no inference API or new training was used. All E3 artifacts are "
              "namespaced under experiment3/ and outputs/experiment3/. Frozen E1/E2 sources and results remain unchanged.", "",
              "## Reproduction", "",
              "The public registration identifies exact inputs, weights, predictions and scoring code. Regenerate only "
              "the report from existing outcomes with the command below. The score command reads saved arrays and also "
              "performs no model forward. Archive any existing generated score/report files before a deliberate rerun "
              "if their original runtime record must be retained.", "",
              "```sh", ".venv/bin/python -m experiment3.score", ".venv/bin/python -m experiment3.report", "```", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    with (ROOT / "primary_conditions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "condition", "endpoint", "mean", "wilson_low", "wilson_high", "minimum", "pass"])
        for s in seeds:
            for name, result in scores[s]["primary_conditions"].items():
                writer.writerow([s, name, result["endpoint"], result["mean"], *result["ci95_wilson"], result["threshold"], result["pass"]])
    print(json.dumps({"report": str(ROOT / "RESULTS.md"), "decisions": decisions,
                      "numeric_forecasts": numeric["all"]["count"], "numeric_misses": numeric["all"]["count"] - numeric["all"]["within_tolerance"]}))


if __name__ == "__main__":
    main()
