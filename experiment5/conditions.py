"""Exhaustive outcome-independent E5 layouts and attention-mask conditions.

Token identity 2*i is Ki; 2*i+1 is Vi. Layout indices also specify canonical
position IDs. Query identity 8 remains last. Condition IDs are local to a
layout shard; the globally unique identity is (panel, layout id, condition id).
"""

import itertools
import json
from pathlib import Path


GROUPED = [0, 2, 4, 6, 1, 3, 5, 7, 8]


def condition(name, kind, keep_keys=None, correct=False, oracle=False, edge=None):
    return {"id": name, "kind": kind, "keep_keys": keep_keys,
            "is_correct_guard": correct, "oracle": oracle, "edge": edge}


def layout_spec(indices, family):
    indices = list(indices)
    if len(indices) == 8:
        indices.append(8)
    pos = {token: index for index, token in enumerate(indices)}
    predecessors = [{token for token in indices[:pos[2*i+1]+1]} for i in range(4)]
    visible_keys = [[j for j in range(4) if 2*j in predecessors[i]] for i in range(4)]
    missing = [sorted(set(range(2*i+2)) - predecessors[i]) for i in range(4)]
    return {"id": "layout_" + "".join(map(str, indices[:8])), "family": family,
            "layout_indices": indices, "visible_keys": visible_keys,
            "restorable_values": [not row for row in missing],
            "missing_native_predecessors": missing,
            "extra_predecessors": [sorted(predecessors[i] - set(range(2*i+2))) for i in range(4)],
            "value_visibility_signature": "|".join("".join(map(str, row)) for row in visible_keys),
            "conditions": []}


def generate_conditions():
    primary, boundary = [], []
    correct = [list(range(i+1)) for i in range(4)]
    for permutation in itertools.permutations(range(8)):
        if any(permutation.index(2*i) > permutation.index(2*i+1) for i in range(4)):
            continue
        is_primary = [t for t in permutation if t % 2] == [1, 3, 5, 7]
        layout = layout_spec(permutation, "primary" if is_primary else "boundary")
        if is_primary:
            alternatives = [[list(c) for c in itertools.combinations(row, i+1) if i in c]
                            for i, row in enumerate(layout["visible_keys"])]
            guards = []
            for rows in itertools.product(*alternatives):
                keep = [list(row) for row in rows]
                is_correct = keep == correct
                name = "correct" if is_correct else "sham_" + "_".join("".join(map(str, row)) for row in keep)
                guards.append(condition(name, "keyguard", keep, correct=is_correct))
            layout["conditions"] = [condition("unguarded", "unguarded"), condition("noop", "noop"),
                                    condition("oracle", "keyguard", correct, correct=True, oracle=True)] + guards
            primary.append(layout)
        else:
            layout["conditions"] = [condition("unguarded", "unguarded"),
                                    condition("keyguard", "keyguard", [[j for j in row if j <= i]
                                                                        for i, row in enumerate(layout["visible_keys"])]),
                                    condition("logical_prefix", "logical_prefix"),
                                    condition("own_key_self", "own_key_self")]
            boundary.append(layout)
    edge = layout_spec(GROUPED, "edge_panel")
    for i in range(4):
        for j in range(i+1, 4):
            keep = [row.copy() for row in correct]
            keep[i].append(j)
            keep[i].sort()
            edge["conditions"].append(condition(f"edge_v{i}_k{j}", "keyguard", keep, edge=[i, j]))
    references = layout_spec(range(9), "references")
    references["conditions"] = [condition("native", "unguarded"), condition("native_own_key_self", "own_key_self")]
    return {"schema_version": 1, "primary": primary, "boundary": boundary,
            "edge_panel": edge, "references": references,
            "counts": {"primary_layouts": len(primary), "primary_cells": sum(len(l["conditions"]) for l in primary),
                       "boundary_layouts": len(boundary), "boundary_cells": sum(len(l["conditions"]) for l in boundary),
                       "edge_cells": 6, "reference_cells": 2}}


def main():
    path = Path("experiment5/conditions.json")
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    grid = generate_conditions()
    path.write_text(json.dumps(grid, indent=2) + "\n")
    print(json.dumps(grid["counts"]))


if __name__ == "__main__":
    main()
