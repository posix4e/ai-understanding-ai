"""Pure scalar arithmetic only; this is not a raw-archive admission auditor."""
from __future__ import annotations

import math
from collections import Counter

from protocol import CONDITIONS, validate_schedule


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected finite numeric value, excluding bool")
    return float(value)


def score_eight(scores, target):
    if not isinstance(scores, (list, tuple)) or len(scores) != 8:
        raise ValueError("Expected exactly eight ordered label logits")
    if type(target) is not int or not 0 <= target < 8:
        raise ValueError("Invalid target")
    values = [finite(v) for v in scores]
    winners = [i for i, value in enumerate(values) if value == max(values)]
    other = [i for i in range(8) if i != target]
    # Difference first; overflow is structural, never converted into a score.
    contrasts = {str(i): finite(values[target] - values[i]) for i in other}
    j = finite(math.fsum(contrasts.values()) / 7)
    choice = winners[0] if len(winners) == 1 else None
    return {"scores": values, "target": target, "J": j, "target_other": contrasts,
            "best_other_margin": min(contrasts.values()), "winners": winners,
            "choice": choice, "tie": choice is None, "correct": choice == target}


def summary(values):
    values = [finite(v) for v in values]
    if not values:
        raise ValueError("Empty aggregation")
    signs = Counter("positive" if v > 0 else "negative" if v < 0 else "zero" for v in values)
    return {"n": len(values), "mean": finite(math.fsum(values) / len(values)),
            **{key: signs[key] for key in ("positive", "negative", "zero")}}


def evaluate_scores(plan, records):
    """Records are {id, scores:[8]}; complete fixed panel or ValueError.

    No raw tensor, frozen-state, sham, resource or lifecycle claims are made.
    A future collector must independently admit those before using this helper.
    """
    validate_schedule(plan["cells"], plan["schedule"])
    if not isinstance(records, list) or len(records) != 768:
        raise ValueError("Require all 768 records before scoring")
    by_id = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != {"id", "scores"}:
            raise ValueError("Unexpected score record fields")
        if type(row["id"]) is not str or row["id"] in by_id:
            raise ValueError("Duplicate/invalid record ID")
        by_id[row["id"]] = row["scores"]
    if set(by_id) != {row["id"] for row in plan["schedule"]}:
        raise ValueError("Missing/unexpected record")
    # Validate every finite vector before any aggregation.
    scored = [dict(case, **score_eight(by_id[case["id"]], case["target"]))
              for case in plan["schedule"]]
    cells = []
    for i, cell in enumerate(plan["cells"]):
        rows = scored[i * 8:(i + 1) * 8]
        j = {(r["strength"], r["arm"]): r["J"] for r in rows}
        r_small = finite(j["1/16", "N"] - j["1/16", "F0"])
        r_full = finite(j["1", "N"] - j["1", "F0"])
        effects = {
            "route_1_16": r_small, "route_1": r_full,
            "route_full_minus_attenuated": finite(r_full - r_small),
            "strength_full_minus_attenuated_native": finite(j["1", "N"] - j["1/16", "N"]),
            "strength_full_minus_attenuated_fixed_B": finite(j["1", "F0"] - j["1/16", "F0"]),
            **{f"sham_S_minus_N_{strength.replace('/', '_')}": finite(j[strength, "S"] - j[strength, "N"])
               for strength in ("0", "1/16", "1")},
        }
        cells.append({"cell_id": cell["cell_id"], "world_id": cell["world_id"], "effects": effects})
    conditions = {}
    for strength, arm in CONDITIONS:
        rows = [r for r in scored if r["strength"] == strength and r["arm"] == arm]
        correct = sum(r["correct"] for r in rows)
        conditions[f"{strength}:{arm}"] = {"n": len(rows), "correct": correct,
            "accuracy": correct / len(rows), "ties": sum(r["tie"] for r in rows),
            "J": summary([r["J"] for r in rows])}
    def effects_for(items):
        return {key: summary([cell["effects"][key] for cell in items]) for key in cells[0]["effects"]}
    return {"status": "DESIGN_ARITHMETIC_ONLY_NOT_ARCHIVE_VALIDATION",
            "implementation_admitted": False, "scientific_success_claim": False,
            "conditions": conditions, "rows": scored, "cells": cells,
            "effects": effects_for(cells),
            "per_world_effects": {world: effects_for([c for c in cells if c["world_id"] == world])
                                  for world in sorted({c["world_id"] for c in cells})}}
