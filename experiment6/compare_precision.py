"""Compare saved original/repeat outcomes without inference or threshold edits."""
from pathlib import Path
import csv
import json

import numpy as np


def main():
    root = Path("outputs/experiment6")
    repeat_root = root / "precision_repeat"
    first = json.loads((root / "confirmatory/scores.json").read_text())
    repeat = json.loads((repeat_root / "confirmatory/scores.json").read_text())
    metrics = ["restricted_accuracy", "unrestricted_accuracy", "raw_target_probability",
               "conditional_target_probability", "candidate_probability_mass"]
    rows = []
    for name in first["conditions"]:
        with np.load(root / "confirmatory" / (name+".npz"), allow_pickle=False) as a, np.load(repeat_root / "confirmatory" / (name+".npz"), allow_pickle=False) as b:
            for field in ("target_indices", "target_token_ids", "query_pair"):
                assert np.array_equal(a[field], b[field])
            row = {"condition": name,
                   "changed_restricted_predictions": int(np.sum(a["candidate_probs"].argmax(1) != b["candidate_probs"].argmax(1))),
                   "changed_unrestricted_predictions": int(np.sum(a["top_token_ids"] != b["top_token_ids"])),
                   "maximum_raw_candidate_probability_difference": float(np.max(np.abs(a["candidate_probs"]-b["candidate_probs"])))}
        for metric in metrics:
            row[metric+"_original"] = first["conditions"][name][metric]["mean"]
            row[metric+"_repeat"] = repeat["conditions"][name][metric]["mean"]
            row[metric+"_change"] = row[metric+"_repeat"]-row[metric+"_original"]
        rows.append(row)
    result = {"original_classification": first["classification"], "repeat_classification": repeat["classification"],
              "same_cases_reused": True, "independent_replication": False,
              "original_validity": first["runner_validity"], "repeat_validity": repeat["runner_validity"],
              "conditions": rows, "maximum_raw_probability_difference": max(r["maximum_raw_candidate_probability_difference"] for r in rows),
              "total_changed_restricted_predictions": sum(r["changed_restricted_predictions"] for r in rows),
              "total_changed_unrestricted_predictions": sum(r["changed_unrestricted_predictions"] for r in rows)}
    (repeat_root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (repeat_root / "comparison.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ("conditions","original_validity","repeat_validity")}))


if __name__ == "__main__":
    main()
