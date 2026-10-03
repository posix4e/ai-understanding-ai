"""Execute the publicly frozen GPT-2 intervention grid, retaining all results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from experiment6.conditions import CONDITIONS, construct_mask
from experiment6.data import VALUES, answer_ids as tokenize_answer_ids, encode_case
from experiment6.model_adapter import GPT2Adapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiment6"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode()); h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output-dir", default="outputs/experiment6/confirmatory")
    args = parser.parse_args()
    manifest_bytes = (OUT / "confirmation_manifest.json").read_bytes()
    committed_manifest = subprocess.check_output([
        "git", "show", f"{args.registration}:outputs/experiment6/confirmation_manifest.json"], cwd=ROOT)
    assert manifest_bytes == committed_manifest, "registration manifest differs from the public commit"
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest["files"].items():
        local = (ROOT / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{args.registration}:{name}"], cwd=ROOT)
        assert local == committed and hashlib.sha256(local).hexdigest() == expected, name
    for name, record in json.loads((OUT / "model_manifest.json").read_text())["files"].items():
        assert sha(ROOT / "work/models/gpt2" / name) == record["sha256"], name
    target = ROOT / args.output_dir
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("output directory is not empty; never overwrite or silently restart a run")
    target.mkdir(parents=True, exist_ok=True)
    (target / "batches").mkdir()
    protocol = json.loads((ROOT / "experiment6/protocol.json").read_text())
    assert protocol["conditions"] == CONDITIONS
    cases = json.loads((ROOT / protocol["inputs_path"]).read_text())
    assert len(cases) == protocol["n"]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "work/models/gpt2", local_files_only=True)
    assert protocol["values"] == VALUES
    assert protocol["candidate_token_ids"] == tokenize_answer_ids(tokenizer)
    encoded = [encode_case(tokenizer, c, protocol["template"]) for c in cases]
    assert len({x.ids.numel() for x in encoded}) == 1
    length = encoded[0].ids.numel()
    # All fragments have identical token lengths, so masks/permutations are shared.
    assert all(e.chunks == encoded[0].chunks and e.suffix == encoded[0].suffix and e.prefix == encoded[0].prefix for e in encoded)
    permutations = {c["id"]: encoded[0].permutation(c["grouped"]) for c in CONDITIONS}
    masks = {c["id"]: {layer: construct_mask(encoded[0], c) for layer in c["layers"]} for c in CONDITIONS}
    adapter = GPT2Adapter.from_local(ROOT / "work/models/gpt2")
    assert sum(p.numel() for p in adapter.model.parameters()) == protocol["parameter_count"]
    before = parameter_hash(adapter.model)
    started = time.perf_counter()
    write_json(target / "started.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
               "registration": args.registration, "n": len(cases), "conditions": len(CONDITIONS),
               "parameters_sha256": before, "device": "cpu", "dtype": "float32",
               "probability_dtype": "float64", "threads": torch.get_num_threads()})
    ids = torch.stack([e.ids for e in encoded])
    answer_ids = torch.tensor(protocol["candidate_token_ids"])
    target_indices = np.array([VALUES.index(c["values"][c["query_pair"]]) for c in cases], dtype=np.int64)
    query_pair = np.array([c["query_pair"] for c in cases], dtype=np.int64)
    target_token_ids = answer_ids.numpy()[target_indices]
    values = [t for i in range(4) for t in encoded[0].chunks[2*i+1]]
    keys = [t for i in range(4) for t in encoded[0].chunks[2*i]]
    suffix = encoded[0].suffix
    raw = {c["id"]: {"candidate_probs": [], "top_token_ids": []} for c in CONDITIONS}
    diagnostics = {"noop_probability_error": 0., "guard_value_state_error": 0.,
                   "query_suffix_state_error": 0., "unedited_key_state_error": 0.}
    ledger = []
    for begin in range(0, len(cases), protocol["batch_size"]):
        batch = ids[begin:begin+protocol["batch_size"]]
        native_state = canonical_state = canonical_probs = None
        chunk = {}
        for condition in CONDITIONS:
            name = condition["id"]
            permutation = permutations[name]
            inputs = batch[:, permutation]
            positions = (permutation if condition["position_mode"] == "canonical" else torch.arange(length)).expand(len(batch), -1)
            result = adapter(inputs, position_ids=positions, layer_masks=masks[name], capture_first_block=True)
            # Retain float32 model computation; normalize the large vocabulary
            # in float64 to avoid the measured native softmax summation error.
            probabilities = result.logits.double().softmax(-1)
            assert bool(torch.isfinite(probabilities).all()) and bool(torch.isfinite(result.first_block).all()), name
            assert bool(torch.allclose(probabilities.sum(-1), torch.ones(len(batch), dtype=torch.float64), atol=1e-6, rtol=0))
            state = result.first_block[:, torch.argsort(permutation)]
            selected = probabilities[:, answer_ids].numpy().copy()
            top = result.logits.argmax(-1).numpy().copy()
            raw[name]["candidate_probs"].append(selected)
            raw[name]["top_token_ids"].append(top)
            chunk[name + "__candidate_probs"] = selected
            chunk[name + "__top_token_ids"] = top
            if name == "native":
                native_state = state.clone()
            if name == "grouped_canonical":
                canonical_state, canonical_probs = state.clone(), probabilities.clone()
            if condition["position_mode"] == "canonical":
                error = float((state[:, suffix] - native_state[:, suffix]).abs().max())
                diagnostics["query_suffix_state_error"] = max(diagnostics["query_suffix_state_error"], error)
                if condition["layers"]:
                    error = float((state[:, keys] - canonical_state[:, keys]).abs().max())
                    diagnostics["unedited_key_state_error"] = max(diagnostics["unedited_key_state_error"], error)
            if name == "guard_block0":
                error = float((state[:, values] - native_state[:, values]).abs().max())
                diagnostics["guard_value_state_error"] = max(diagnostics["guard_value_state_error"], error)
            if name == "grouped_noop":
                error = float((probabilities - canonical_probs).abs().max())
                diagnostics["noop_probability_error"] = max(diagnostics["noop_probability_error"], error)
        chunk["case_indices"] = np.arange(begin, begin+len(batch))
        chunk_path = target / "batches" / f"batch_{begin:04d}.npz"
        np.savez_compressed(chunk_path, **chunk)
        ledger.append({"file": str(chunk_path.relative_to(target)), "sha256": sha(chunk_path),
                       "begin": begin, "end": begin+len(batch)})
        write_json(target / "ledger.json", ledger)
        if begin % 64 == 0:
            print(json.dumps({"completed_cases": begin+len(batch), "total_cases": len(cases),
                              "seconds": time.perf_counter()-started}), flush=True)
    for name, data in raw.items():
        np.savez_compressed(target / (name + ".npz"),
            candidate_probs=np.concatenate(data["candidate_probs"]),
            top_token_ids=np.concatenate(data["top_token_ids"]),
            target_indices=target_indices, target_token_ids=target_token_ids, query_pair=query_pair)
    after = parameter_hash(adapter.model)
    checks = {"complete_finite_outputs": len(ledger)*protocol["batch_size"] == len(cases),
              "parameters_unchanged": before == after,
              "noop_probabilities": diagnostics["noop_probability_error"] <= protocol["noop_probability_tolerance"],
              "guard_first_block_values": diagnostics["guard_value_state_error"] <= protocol["state_tolerance"],
              "first_block_query_suffix": diagnostics["query_suffix_state_error"] <= protocol["state_tolerance"],
              "unedited_first_block_keys": diagnostics["unedited_key_state_error"] <= protocol["state_tolerance"]}
    write_json(target / "validity.json", {"all_pass": all(checks.values()), "checks": checks,
               "diagnostics": diagnostics, "parameters_before_sha256": before, "parameters_after_sha256": after})
    write_json(target / "finished.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
               "registration": args.registration, "n": len(cases), "conditions": len(CONDITIONS),
               "seconds": time.perf_counter()-started, "validity_pass": all(checks.values()),
               "files": {p.name: sha(p) for p in sorted(target.glob("*.npz"))}})
    print(json.dumps({"finished": True, "validity": checks, "seconds": time.perf_counter()-started}), flush=True)


if __name__ == "__main__":
    main()
