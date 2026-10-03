"""Generate E5 report exclusively from completed saved scores and provenance."""

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path("outputs/experiment5")


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def interval(value):
    low, high = value["ci95"]
    return f"{value['mean']:.4f} [{low:.4f}, {high:.4f}]"


def yes(value):
    return "pass" if value else "FAIL"


def render(scores, decisions, numeric, protocol, lock, start, finish):
    seeds = [str(s) for s in protocol["model_seeds"]]
    if set(scores) != set(seeds):
        raise ValueError("Report requires every registered seed")
    expected = {"all_six_valid": all(all(scores[s]["validity_gates"].values()) for s in seeds),
                "all_six_recovery": all(scores[s]["primary_gates"]["every_layout_query_accuracy"] and scores[s]["primary_gates"]["native_minus_correct_upper"] for s in seeds),
                "all_six_specificity": all(scores[s]["primary_gates"]["correct_minus_mean_shams_lower"] for s in seeds)}
    expected["primary_claim_pass"] = all(expected.values())
    expected["secondary"] = {name: expected["all_six_valid"] and all(scores[s]["secondary_gates"][name] for s in seeds)
                             for name in ("boundary_logical_prefix", "boundary_own_key_self", "edge_redirection")}
    if decisions != expected or start["preregistration_commit"] != lock["commit"]:
        raise ValueError("Decisions or public registration inconsistent with scored records")
    rows = []
    for s in seeds:
        result = scores[s]
        a = result["aggregate"]
        failed = [key for group in ("validity_gates", "primary_gates") for key, passed in result[group].items() if not passed]
        rows.append([s, f"{a['native_accuracy']:.4f}", f"{a['primary_correct_accuracy']:.4f}",
                     f"{a['primary_correct_min_layout_query_accuracy']:.4f}", interval(result["comparisons"]["native_minus_correct"]),
                     interval(result["comparisons"]["correct_minus_mean_shams"]), ", ".join(failed) or "none"])
    verdict = "passed" if decisions["primary_claim_pass"] else "did not pass"
    lines = ["# Experiment 5: causal visibility across all restorable serializations", "",
             f"**The full prespecified six-seed primary claim {verdict}.**", "",
             "Recovery, control specificity, implementation validity, secondary findings, and numerical forecasts are separate results. "
             "An algebraic identity check or secondary success cannot rescue a primary failure.", "",
             table(["Decision", "Result"], [[key, yes(value)] for key, value in decisions.items() if key != "secondary"]), "",
             "## Primary result: every fresh model", "",
             table(["Seed", "Native accuracy", "Correct-guard accuracy", "Worst layout/query accuracy", "Native−guard P [95% CI]",
                    "Guard−mean shams P [95% CI]", "Failed validity/primary gates"], rows), "",
             "Recovery requires ≥0.95 accuracy in every one of 105×4 layout/query cells per seed, and an upper paired 95% "
             "bound on native-minus-guard probability ≤0.01. Specificity requires the lower bound on guard-minus-mean-shams "
             "probability strictly >0.02. Every gate must pass in all six seeds. The native-minus-guard contrast is signed; "
             "it is a noninferiority criterion rather than a bound on absolute probability differences.", "",
             "## Complete failures and controls", "",
             "[Every wrong primary prediction](primary_failure_cases.csv), including mistakes within passing cells, "
             "[every failing primary layout/query cell](primary_failures.csv), [every condition mean](condition_results.csv), "
             "and [every query-stratified result](per_query_results.csv) are retained. The condition and query tables include "
             "all references, oracles, shams, boundary layouts, and edge tests; full softmax distributions remain in the raw shards.", "",
             table(["Seed", "Failed primary cells", "Worst layout", "Worst query", "Accuracy"],
                   [[s, scores[s]["primary_failed_cells"], scores[s]["worst_primary_cells"][0]["layout"],
                     scores[s]["worst_primary_cells"][0]["query_pair"], f"{scores[s]['worst_primary_cells'][0]['accuracy']:.4f}"] for s in seeds]), "",
             "[All control summaries](controls_summary.csv) show the correct mask, mean of each layout's complete sham family, "
             "and best individual sham by point estimate. There are 624 shams across 102 eligible layouts. Controls are averaged "
             "within each layout, then layouts receive equal weight. Three layouts admit no alternative that preserves the own "
             "key and row degree; they remain recovery tests but are excluded from specificity. Best-sham points are descriptive "
             "and have no selected-best confidence interval. Success against the mean does not establish superiority to every sham.", "",
             "## Separate secondary results", "",
             table(["Secondary all-six-seed decision", "Result"], [[name, yes(value)] for name, value in decisions["secondary"].items()]), "",
             table(["Seed", "Boundary prefix accuracy [95% CI]", "Boundary own-key/self accuracy [95% CI]", "Prefix cells ≥.95", "Own-key/self cells ≥.95", "Edge redirection [95% CI]"],
                   [[s, interval(scores[s]["boundary"]["logical_prefix"]["accuracy"]), interval(scores[s]["boundary"]["own_key_self"]["accuracy"]),
                     f"{scores[s]['boundary']['logical_prefix']['fraction_layout_query_accuracy_ge_0_95']:.4f}",
                     f"{scores[s]['boundary']['own_key_self']['fraction_layout_query_accuracy_ge_0_95']:.4f}", interval(scores[s]["edge"])] for s in seeds]), "",
             "The two boundary accuracy gates require the pooled equal-layout accuracy ≥0.95 in every seed. The fraction of "
             "individual layout/query cells above .95 is descriptive, not an additional gate. Boundary queries have 64 cases "
             "per stratum. The edge gate requires a lower paired 95% bound strictly >0.05 in every seed.", "",
             table(["Seed", "All 2,415 boundary layouts: policy", "Target probability [95% CI]", "Accuracy [95% CI]"],
                   [[s, name, interval(scores[s]["boundary"][name]["probability"]), interval(scores[s]["boundary"][name]["accuracy"])]
                    for s in seeds for name in ("unguarded", "keyguard", "logical_prefix", "own_key_self")]), "",
             "For re-added Vi←Kj edges, the edge statistic uses only queries for Kj. It measures the change in probability of "
             "the wrong value Vi relative to the mean change of the two other displayed wrong values, compared with the grouped "
             "correct guard. Each of six edges receives equal weight after within-query normalization. This comparison controls "
             "for nonspecific redistribution among wrong answers; q0 is not eligible for this aggregate.", "",
             table(["Seed", "Edge", "Eligible-query contrast [95% CI]"], [[s, edge, interval(value)] for s in seeds for edge, value in scores[s]["per_edge"].items()]), "",
             "### Fixed key order and permuted values", "",
             "The fixed-key-order subpanel consists of 23 boundary value orders plus the identity value order from primary "
             "grouping. They are shown separately below; no mixed-sample or missing-condition 24-layout average is manufactured. "
             "Primary grouped references here use the same first 256 rows as the boundary panel.", "",
             table(["Seed", "23 boundary layouts: unguarded accuracy", "23: keyguard accuracy", "23: logical-prefix accuracy", "23: own-key/self accuracy", "Grouped identity: unguarded", "Grouped identity: correct guard"],
                   [[s] + [f"{scores[s]['boundary_fixed_key_order']['conditions'][name]['accuracy']['mean']:.4f}" for name in ("unguarded", "keyguard", "logical_prefix", "own_key_self")]
                    + [f"{scores[s]['grouped_reference_boundary_rows'][name]['accuracy']:.4f}" for name in ("unguarded", "correct")] for s in seeds]), "",
             "## Validity and numerical predictions", "",
             table(["Seed", "Validity", "No-op maximum error", "Correct value-state error", "Query-state error, all cells", "Untouched-key error", "Full-oracle probability error", "Own-key/self value-state error"],
                   [[s, yes(all(scores[s]["validity_gates"].values()))] + [f"{scores[s]['manipulation_maxima'][name]:.3g}" for name in
                     ("noop_full_probability", "correct_l1_value_native", "all_l1_query_native", "all_l1_key_baseline", "oracle_full_probability", "own_key_self_l1_value_reference")] for s in seeds]), "",
             "All saved output cells are checked for complete shapes, finite values, normalized full-class probabilities, "
             "argmax/accuracy/target consistency, and nonnegative state errors. Shard hashes and layout coverage are checked "
             "against completion ledgers. No-op distribution tolerance is 1e-6; registered state/oracle tolerances are 1e-5. "
             "Individual state differences are measured by the frozen runner; independent synthetic tests audit the hook implementation.", "",
             f"The separate 72 aggregate numerical forecasts have MAE **{numeric['mae']:.4f}**, RMSE **{numeric['rmse']:.4f}**, "
             f"and **{numeric['within_tolerance']}/{numeric['count']}** outcomes within ±0.05. "
             "[Every forecast](forecasts_vs_results.csv) and [every miss](forecast_misses.csv) are included. These are judgmental "
             "point forecasts, not predictive intervals, and their scores do not replace any scientific gate.", "",
             "## Interpretation and limits", "",
             "All 2,520 pair-respecting orders keep Ki before Vi. Exactly 105 preserve value order and therefore retain "
             "every native predecessor at every value node. With canonical IDs, removing extra future-logical keys must "
             "restore first-layer value states. Query states are invariant because the final query sees every token. Key "
             "states remain changed; their second-layer keys/values and attention normalization can still alter the answer. "
             "Thus final-answer recovery and the comparison with matched masks are empirical, while value-state and full-oracle "
             "restoration are algebraic manipulation checks. The exact-restoration proof is in [THEORY.md](THEORY.md).", "",
             "The 2,415 remaining orders can omit earlier native keys or values. Deleting future keys alone, deleting all "
             "newly visible future-logical identities, and retaining only the own key plus self are distinct policies. Own-key/self "
             "gives invariant first-layer value states under all layouts, but those are not generally native states. These masks "
             "use known canonical pair identities; success does not demonstrate independent discovery of pair associations.", "",
             "This is an exhaustive finite synthetic study, conditional on a fixed two-layer, four-head, 70,720-parameter "
             "architecture and six fresh training seeds. Canonical position IDs preserve training-slot information. The study "
             "does not establish ordinary position-invariant generalization, performance on naturally pretrained LLMs, minimal "
             "circuits, or a new general context-restoration principle. E1–E4, including negative results, remain preserved.", "",
             "Confidence intervals use 2,000 query-stratified paired dictionary-row resamples, shared across layouts, conditions "
             "and seeds within panels. They describe uncertainty over dictionaries conditional on these checkpoints, not uncertainty "
             "over model training. Wilson intervals in CSVs are individual intervals, not simultaneous coverage. Layouts and "
             "interventions on the same dictionaries are dependent. The boundary uses the first 256 primary dictionaries; a "
             "separate bootstrap seed does not make it an independent dataset.", "",
             "## Prospective record and reproduction", "",
             f"Frozen registration: [{lock['commit']}](https://github.com/posix4e/ai-understanding-ai/commit/{lock['commit']}). "
             f"Public verification: `{lock.get('verified_at_utc', 'see remote_lock.json')}`. "
             f"Confirmation start: `{start['utc']}`; finish: `{finish['utc']}`. "
             f"Report generated: `{datetime.now(timezone.utc).isoformat()}`.", "",
             "Training seeds 6–11 and the unchanged 2,000-step recipe were publicly fixed before training. New inputs contain "
             "1,024 distinct dictionaries with 256 queries per pair; the input audit checks historical exclusions. All inference "
             "ran locally on CPU with four threads; no inference API or new shifted-layout fitting was used. Raw shards, per-seed "
             "completion ledgers, [input audit](input_audit.json), [training cohort](training_cohort.json), and the machine-readable "
             "[scores](confirmatory/scores.json) retain provenance, counts, and failures.", "",
             "To reproduce on a separate checkout, first preserve the published outcome directory and its hashes. The frozen "
             "model-evaluation command refuses to overwrite completed confirmation; interrupted execution uses verified shards "
             "only. Pure postprocessing can be reproduced from copied raw artifacts with:", "", "```sh",
             ".venv/bin/python -m experiment5.scorer", ".venv/bin/python -m experiment5.report", "```", "",
             "Original model evaluation command: `.venv/bin/python -m experiment5.run_phase`; deterministic crash resumption: "
             "`.venv/bin/python -m experiment5.run_phase --resume`. Neither command is executed by this report.", ""]
    return "\n".join(lines)


def main():
    paths = [ROOT / "confirmatory/scores.json", ROOT / "confirmatory/global_decisions.json", ROOT / "confirmatory/forecast_statistics.json",
             Path("experiment5/protocol.json"), ROOT / "remote_lock.json", ROOT / "confirmatory/start.json", ROOT / "confirmatory/finish.json"]
    if any(not path.is_file() for path in paths):
        raise SystemExit("Completed E5 scores and public provenance required")
    output = render(*map(read, paths))
    path = ROOT / "RESULTS.md"
    path.write_text(output)
    print(path)


if __name__ == "__main__":
    main()
