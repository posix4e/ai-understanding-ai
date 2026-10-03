"""The declared native-only calibration. Never evaluates grouped inputs."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoTokenizer

from experiment6.data import TEMPLATES, VALUES, answer_ids, encode_case, generate_cases
from experiment6.model_adapter import GPT2Adapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiment6"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    # Verify exactly the code, inputs and plan published before these forwards.
    manifest = json.loads((OUT / "calibration_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        local = (ROOT / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{args.registration}:{name}"], cwd=ROOT)
        assert local == committed and hashlib.sha256(local).hexdigest() == expected, name
    for name, record in json.loads((OUT / "model_manifest.json").read_text())["files"].items():
        assert hashlib.sha256((ROOT / "work/models/gpt2" / name).read_bytes()).hexdigest() == record["sha256"], name
    result_path = OUT / "calibration_results.json"
    if result_path.exists() or (OUT / "calibration_started.json").exists():
        raise RuntimeError("calibration already started: do not overwrite or silently rerun")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "work/models/gpt2", local_files_only=True)
    cases = json.loads((OUT / "calibration_inputs.json").read_text())
    assert cases == generate_cases(64, 10600001)
    allowed = torch.tensor(answer_ids(tokenizer))
    encoded = {name: [encode_case(tokenizer, c, name) for c in cases] for name in TEMPLATES}
    for items in encoded.values():
        assert len({x.ids.numel() for x in items}) == 1
    start = {"at_utc": datetime.now(timezone.utc).isoformat(),
             "registration": args.registration, "conditions": "native only",
             "n_cases_per_template": 64, "templates": TEMPLATES}
    (OUT / "calibration_started.json").write_text(json.dumps(start, indent=2) + "\n")
    adapter = GPT2Adapter.from_local(ROOT / "work/models/gpt2")
    all_rows, summaries = [], []
    total_start = time.perf_counter()
    for template in TEMPLATES:
        template_start = time.perf_counter()
        template_rows = []
        for begin in range(0, len(cases), 8):
            batch = torch.stack([x.ids for x in encoded[template][begin:begin+8]])
            logits = adapter(batch).logits
            probabilities = logits.softmax(-1)
            candidate_probs = probabilities[:, allowed]
            conditional = candidate_probs / candidate_probs.sum(-1, keepdim=True)
            for j in range(batch.shape[0]):
                idx = begin + j
                case = cases[idx]
                target = case["values"][case["query_pair"]]
                target_index = VALUES.index(target)
                winner = int(logits[j].argmax())
                row = {"template": template, "case": idx, "query_pair": case["query_pair"],
                       "target_value": target,
                       "choice_correct": int(int(conditional[j].argmax()) == target_index),
                       "full_vocab_correct": int(winner == int(allowed[target_index])),
                       "target_probability": float(candidate_probs[j, target_index]),
                       "conditional_target_probability": float(conditional[j, target_index]),
                       "answer_mass": float(candidate_probs[j].sum()),
                       "top_token_id": winner, "top_token": tokenizer.decode([winner])}
                for k, value in enumerate(VALUES):
                    row[f"p_{value}"] = float(candidate_probs[j, k])
                template_rows.append(row)
        count = len(template_rows)
        avg = lambda key: sum(r[key] for r in template_rows) / count
        strata = [sum(r["choice_correct"] for r in template_rows if r["query_pair"] == q) / 16 for q in range(4)]
        summary = {"template": template, "n": count, "tokens": encoded[template][0].ids.numel(),
                   "choice_accuracy": avg("choice_correct"),
                   "full_vocab_accuracy": avg("full_vocab_correct"),
                   "target_probability": avg("target_probability"),
                   "answer_mass": avg("answer_mass"), "query_accuracy": strata,
                   "native_competence": avg("choice_correct") >= .8 and min(strata) >= .7,
                   "seconds": time.perf_counter()-template_start}
        summaries.append(summary)
        all_rows.extend(template_rows)
        print(json.dumps(summary), flush=True)
    selected = max(summaries, key=lambda x: (x["choice_accuracy"], x["target_probability"], -TEMPLATES.index(x["template"])))
    estimate_512 = selected["seconds"] / 64 * 512 * 15
    n_confirm = 256 if estimate_512 > 3600 else 512
    with (OUT / "calibration_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    result = {"registration": args.registration, "summaries": summaries,
              "selected_template": selected["template"], "selected_competent": selected["native_competence"],
              "model_parameters": sum(p.numel() for p in adapter.model.parameters()),
              "estimated_confirmation_seconds_512": estimate_512, "confirmation_n": n_confirm,
              "seconds": time.perf_counter()-total_start,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "reordered_or_repaired_forwards": 0}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_template": selected["template"], "confirmation_n": n_confirm,
                      "adequate_native_competence": selected["native_competence"]}), flush=True)


if __name__ == "__main__":
    main()
