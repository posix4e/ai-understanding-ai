"""Design-only metadata and schedule. No model/runtime/old-runner imports."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATUS = "DESIGN_READY_RUNNER_NOT_IMPLEMENTED"
CENSUS_PATH = "outputs/moe-mechanisms/ORGAN_ROUTE_CENSUS_V1/plan.json"
CENSUS_SHA256 = "133c644bae49d7e0cf30704e5be5f05797ceff022b5763ef358f95335cdfe701"
TAIL_PATH = "outputs/moe-mechanisms/ORGAN_TAIL_ATTRIBUTION_V1/plan.json"
MODEL_REVISION = "0da7a48b0276d500ce5922fd2b33944091fc6c09"
STRENGTHS = ("0", "1/16", "1")
CONDITIONS = (("0", "N"), ("0", "S"), ("1/16", "N"),
              ("1/16", "S"), ("1/16", "F0"),
              ("1", "N"), ("1", "S"), ("1", "F0"))
SOURCE_FILES = ("protocol.py", "metrics.py", "test_protocol.py", "test_metrics.py", "README.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def exact(left, right):
    return canonical(left) == canonical(right)


def metadata_cells(census, tail):
    """Validate all metadata, then derive the fixed 8x3x4 cell census."""
    contexts = census["contexts"]
    expected_contexts = [(s, a) for s in range(8) for a in range(3)]
    require(len(contexts) == 24, "Expected 24 contexts")
    require(exact([(c["state"], c["action"]) for c in contexts], expected_contexts),
            "Context order/coverage differs")
    require([c["id"] for c in contexts] == [f"s{s}-a{a}" for s, a in expected_contexts],
            "Context identity differs")
    require(exact(contexts, tail["contexts"]), "Tail/census context identity differs")
    worlds = tail["worlds"]
    require([w["id"] for w in worlds] == [f"dev{i:02d}" for i in range(4)], "World order differs")
    for w in worlds:
        require(len(w["transitions"]) == 3, "World action count differs")
        for row in w["transitions"]:
            require(len(row) == 8 and all(type(v) is int for v in row)
                    and sorted(row) == list(range(8)), "Invalid fixed transitions")
    require(exact(tail["label_ids"], list(range(34, 42))), "Label IDs differ")
    schedule = census["schedule"]
    require(len(schedule) == 192, "Expected 192 B/W source entries")
    cells = []
    for ci, context in enumerate(contexts):
        for wi, world in enumerate(worlds):
            i = ci * 4 + wi
            cell_id = f"{context['id']}-{world['id']}"
            sources = {}
            for j, arm in enumerate(("B", "W")):
                row = schedule[2 * i + j]
                identity = {"index": 2 * i + j, "id": f"{2*i+j:04d}",
                            "arm": arm, "cell_id": cell_id,
                            "context_id": context["id"], "world_id": world["id"]}
                require(exact({k: row[k] for k in identity}, identity), "Source schedule differs")
                sources[arm] = {
                    "census_id": row["id"],
                    "census_call_path": f"outputs/moe-mechanisms/ORGAN_ROUTE_CENSUS_V1/calls/{row['id']}.npz",
                    "census_metadata_path": f"outputs/moe-mechanisms/ORGAN_ROUTE_CENSUS_V1/calls/{row['id']}.json",
                    "tail_source": copy.deepcopy(row),
                }
            cells.append({"index": i, "cell_id": cell_id, "context_id": context["id"],
                          "world_id": world["id"], "state": context["state"],
                          "action": context["action"],
                          "target": world["transitions"][context["action"]][context["state"]],
                          "sources": sources})
    return cells


def make_schedule(cells):
    require(len(cells) == 96, "Expected all 96 cells")
    require(len({c["cell_id"] for c in cells}) == 96, "Duplicate cells")
    schedule = []
    for i, cell in enumerate(cells):
        require(type(cell["index"]) is int and cell["index"] == i, "Cell order differs")
        require(type(cell["target"]) is int and 0 <= cell["target"] < 8, "Invalid target")
        native_ids = {}
        for strength, arm in CONDITIONS:
            index = len(schedule)
            row_id = f"{index:04d}"
            if arm == "N":
                native_ids[strength] = row_id
            schedule.append({
                "index": index, "id": row_id, "cell_id": cell["cell_id"],
                "context_id": cell["context_id"], "world_id": cell["world_id"],
                "strength": strength, "arm": arm, "target": cell["target"],
                "B_census_id": cell["sources"]["B"]["census_id"],
                "W_census_id": cell["sources"]["W"]["census_id"],
                "same_strength_native_id": native_ids[strength],
                "route_source": None if arm == "N" else (
                    {"kind": "same_condition_native", "id": native_ids[strength]} if arm == "S"
                    else {"kind": "frozen_census_B", "id": cell["sources"]["B"]["census_id"]}),
            })
    return schedule


def validate_schedule(cells, schedule):
    require(exact(schedule, make_schedule(cells)), "Schedule missing, duplicated, reordered or altered")
    return {"cells": 96, "suffixes": 768, "native": 288, "self_shams": 288, "fixed_B_routes": 192}


def build_plan(root=ROOT):
    """Read two hash-bound JSON plans only; never opens scientific tensor files."""
    root = Path(root)
    require(digest(root / CENSUS_PATH) == CENSUS_SHA256, "Census plan hash differs")
    census = json.loads((root / CENSUS_PATH).read_text())
    tail_sha = census["lineage_sha256"][TAIL_PATH]
    require(digest(root / TAIL_PATH) == tail_sha, "Tail plan hash differs")
    tail = json.loads((root / TAIL_PATH).read_text())
    require(Path(census["model_path"]).name == MODEL_REVISION, "Model revision differs")
    cells = metadata_cells(census, tail)
    schedule = make_schedule(cells)
    return {
        "schema": "organ-route-strength-design-v1", "status": STATUS,
        "registered": False, "runner_implemented": False, "scientific_execution_count": 0,
        "source_plan_sha256": {CENSUS_PATH: CENSUS_SHA256, TAIL_PATH: tail_sha},
        "source_sha256": {str((HERE / name).relative_to(ROOT)): digest(HERE / name) for name in SOURCE_FILES},
        "model": {"id": "ibm-granite/granite-3.1-1b-a400m-instruct", "revision": MODEL_REVISION,
                  "path": census["model_path"], "model_files_sha256": census["model_files"],
                  "installed_model_source_sha256": census["installed_model_source_sha256"]},
        "runtime": {k: census["settings"][k] for k in
                    ("python", "torch", "transformers", "numpy", "dtype", "device", "threads",
                     "batch_size", "attention", "cache", "logits_scaling")},
        "resource": {"timeout_seconds": 900, "max_rss_bytes": 10 * 2**30,
                     "max_mps_driver_bytes": 10 * 2**30, "max_archive_bytes": 2 * 2**30,
                     "min_free_bytes": 8 * 2**30, "preflight_required_before_registration": True},
        "label_ids": tail["label_ids"], "strengths": list(STRENGTHS),
        "cells": cells, "schedule": schedule, "counts": validate_schedule(cells, schedule),
        "work": {"suffixes": 768, "decoder_blocks": 3072, "full_host_forwards": 0,
                 "prefix_forwards": 0, "reader_calls": 0, "energy_calls": 0, "writer_calls": 0,
                 "fits": 0, "backwards": 0, "optimizer_steps": 0, "tokenizer_calls": 0,
                 "provider_calls": 0, "generated_tokens": 0},
        "interpolation": {
            "boundary": "Original common pre-writer full hidden tensor; edit only final row.",
            "operation": "delta_host(lambda)=BF16(FP32(lambda)*saved_delta32); returned_final=native_BF16_add(hidden_before_final,delta_host(lambda)). Other rows copied unchanged.",
            "endpoints": "Strength 0 must match saved B returned bytes; strength 1 must match saved W returned bytes, with no tolerance. A failed endpoint closes admission; no outcome-dependent recipe repair.",
            "intermediate": "Exactly 1/16, prospectively fixed. FP32 multiplication then BF16 cast then original BF16 addition; no normalization, adaptive gain, probability refit, or interpolation of rounded endpoint hidden states.",
        },
        "routing": "N recomputes native policy. S replays the same-strength N tuple. F0 replays frozen B policy. Replay all four layers and the full sequence: IDs, gates and grouping. Experts always recompute on current hidden inputs; grouped-kernel numerical effects belong to the intervention.",
        "admission": [
            "Complete all 768 finite calls and unchanged state/resource/lifecycle checks before scoring.",
            "All 288 same-condition full-logit/internal-trace shams must be exact, and all 192 native endpoint traces must match the historical B/W census.",
            "Validate actual delivered tuples against their correct donor; never compare clamped policy to current native logits as if it were native top-k.",
            "No cross-policy nonfinal equality gate; retain full nonfinal drift instead.",
        ],
        "metrics": {"primary": "J=target logit minus mean of the other seven fixed-label logits.",
                    "route_effect": "R_lambda=J(N_lambda)-J(F0_lambda); positive helps the registered target.",
                    "amplitude_contrast": "R_1-R_1/16; positive means larger route contribution at full strength.",
                    "secondary": "All seven target-other contrasts, best-other margin, strict unique choice/correctness/ties, norms, support/gate changes, and sham differences; no cell filtering."},
        "interpretation": "Descriptive whole-routing-policy contrasts over 96 exposed cells and only 24 contexts. No route-ID-only, natural mediation, semantic composition, energy advantage, novelty, population or p-value claim. Three strengths do not resolve a continuous dose-response curve.",
        "pending_before_execution": ["Runner and owner implementation", "Independent full-tensor/provenance audit",
            "Exact census raw archive seals", "All endpoint/sham admission code", "Resource and storage preflight",
            "Source review and execution registration"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-design", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan()
    with args.write_design.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": STATUS, "path": str(args.write_design), "sha256": digest(args.write_design)}))
