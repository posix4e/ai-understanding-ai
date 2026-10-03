"""Post-run export of the reused-input float64 repeat; no model inference."""
import csv
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main():
    root = Path("outputs/experiment6/precision_repeat")
    scores = json.loads((root / "confirmatory/scores.json").read_text())
    # Correct the unchanged scorer's generic narrative for the reused sample.
    # Numerical scores and all original-run files remain untouched.
    report_path = root / "RESULTS.md"
    report = report_path.read_text()
    report = report.replace("# E6: pretrained GPT-2 confirmation", "# E6: GPT-2 precision repeat on reused inputs")
    report = report.replace("512 fresh dictionaries; one pretrained checkpoint.",
        "The same 512 dictionaries as the original run; one pretrained checkpoint. "
        "This is an outcome-informed numerical repeat, not independent confirmation.")
    report_path.write_text(report)
    protocol = json.loads(Path("experiment6/precision_protocol.json").read_text())
    forecasts = json.loads(Path("experiment6/predictions.json").read_text())
    cases = json.loads(Path(protocol["inputs_path"]).read_text())
    values = protocol["values"]
    tokenizer = AutoTokenizer.from_pretrained("work/models/gpt2", local_files_only=True)
    rows = []
    for condition in scores["conditions"]:
        with np.load(root / "confirmatory" / (condition + ".npz"), allow_pickle=False) as raw:
            probs = raw["candidate_probs"]
            for i, case in enumerate(cases):
                target = int(raw["target_indices"][i]); winner = int(probs[i].argmax())
                top = int(raw["top_token_ids"][i]); mass = float(probs[i].sum())
                rows.append({"condition": condition, "case": i, "query_pair": case["query_pair"],
                    "query_label": case["keys"][case["query_pair"]],
                    "table": "; ".join(f"{k}={v}" for k, v in zip(case["keys"], case["values"])),
                    "target_value": values[target], "restricted_prediction": values[winner],
                    "restricted_correct": int(winner == target), "top_token_id": top,
                    "top_token_text": tokenizer.decode([top]),
                    "unrestricted_correct": int(top == int(raw["target_token_ids"][i])),
                    "raw_target_probability": float(probs[i, target]), "answer_probability_mass": mass,
                    "conditional_target_probability": float(probs[i, target]) / mass})
    with (root / "all_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    actual = {key: scores["conditions"][condition]["restricted_accuracy"]["mean"]
              for key, condition in [("native_accuracy", "native"),
                 ("grouped_canonical_accuracy", "grouped_canonical"),
                 ("guard_block0_accuracy", "guard_block0"),
                 ("guard_block5_accuracy", "guard_block5"), ("guard_all_accuracy", "guard_all")]}
    actual["guard_minus_mean_shams_conditional_target_probability"] = scores["contrasts"]["specificity_guard_minus_mean_shams_conditional_probability"]["mean"]
    forecast_rows = []
    for name, predicted in forecasts["expected_aggregate"].items():
        error = abs(actual[name]-predicted)
        forecast_rows.append({"endpoint": name, "predicted": predicted, "observed": actual[name],
                              "absolute_error": error, "within_tolerance": error <= forecasts["absolute_error_tolerance"]})
    with (root / "forecasts_vs_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(forecast_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(forecast_rows)
    summary = {"forecasts": forecast_rows, "within_tolerance": sum(r["within_tolerance"] for r in forecast_rows),
               "total": len(forecast_rows), "mae": sum(r["absolute_error"] for r in forecast_rows)/len(forecast_rows),
               "absolute_tolerance": forecasts["absolute_error_tolerance"],
               "descriptive_only": True, "same_cases_reused": True, "independent_replication": False}
    (root / "forecast_scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"prediction_rows": len(rows), "forecast_hits": summary["within_tolerance"], "forecast_count": len(forecast_rows)}))


if __name__ == "__main__":
    main()
