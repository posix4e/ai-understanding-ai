"""Enumerate all 24 slot permutations without model or outcome access.

Permutation pi maps ORIGINAL PAIR index i to assigned slot pi[i]. Presented
grouped order is K0 K1 K2 K3 V0 V1 V2 V3 Q. A coherent map assigns positions
2*pi[i] to Ki and 2*pi[i]+1 to Vi. A value-only map keeps Ki at 2*i but assigns
Vi to 2*pi[i]+1. Therefore value-only slot prediction is V[inverse_pi(q)].
"""

import itertools
import json
from pathlib import Path


IDENTITY = [0, 1, 2, 3]
INTERLEAVED = list(range(9))
GROUPED = [0, 2, 4, 6, 1, 3, 5, 7, 8]


def generate_conditions():
    def spec(name, family, layout, position_ids, permutation=None,
             slot_value_by_query=None, slot_key_by_value=None, defined=True,
             primary_group=None):
        permutation = IDENTITY[:] if permutation is None else list(permutation)
        derangement = all(i != value for i, value in enumerate(permutation))
        return {"id": name, "family": family, "layout_indices": list(layout),
                "position_ids": position_ids, "permutation": permutation,
                "slot_value_by_query": IDENTITY[:] if slot_value_by_query is None else list(slot_value_by_query),
                "slot_key_by_value": IDENTITY[:] if slot_key_by_value is None else list(slot_key_by_value),
                "slot_mapping_defined": defined, "derangement": derangement,
                "primary_group": primary_group}

    conditions = [spec("original_native", "reference", INTERLEAVED, None),
                  spec("original_noop", "reference", INTERLEAVED, INTERLEAVED[:]),
                  spec("grouped_native", "reference", GROUPED, None, defined=False),
                  spec("grouped_canonical", "reference", GROUPED, GROUPED[:])]
    for permutation in itertools.permutations(range(4)):
        pi = list(permutation)
        if pi == IDENTITY:
            continue
        suffix = "".join(map(str, pi))
        derangement = all(i != value for i, value in enumerate(pi))
        inverse = [pi.index(i) for i in range(4)]
        conditions.append(spec("coherent_" + suffix, "coherent", GROUPED,
                               [2 * i for i in pi] + [2 * i + 1 for i in pi] + [8], pi,
                               primary_group="coherent_control" if derangement else None))
        conditions.append(spec("value_" + suffix, "value_only", GROUPED,
                               [0, 2, 4, 6] + [2 * i + 1 for i in pi] + [8], pi,
                               slot_value_by_query=inverse, slot_key_by_value=pi,
                               primary_group="value_derangement" if derangement else None))
    return conditions


def main():
    path = Path("experiment3/conditions.json")
    if path.exists():
        raise SystemExit(f"Refusing to replace {path}")
    conditions = generate_conditions()
    path.write_text(json.dumps(conditions, indent=2) + "\n")
    print(json.dumps({"path": str(path), "conditions": len(conditions),
                      "primary_value_derangements": sum(c["primary_group"] == "value_derangement" for c in conditions),
                      "primary_coherent_controls": sum(c["primary_group"] == "coherent_control" for c in conditions)}))


if __name__ == "__main__":
    main()
