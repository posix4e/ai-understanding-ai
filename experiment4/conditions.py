"""Outcome-independent E4 grid: nine row-degree-matched masks plus references."""

import json
from pathlib import Path


GROUPED = [0, 2, 4, 6, 1, 3, 5, 7, 8]


def keep_keys(a, b):
    """For Vi retain i+1 keys, including Ki. Correct training prefix: a=0,b=3."""
    if a not in (0, 2, 3) or b not in (0, 1, 3):
        raise ValueError("Invalid matched-guard coordinates")
    return [[0], sorted([1, a]), [key for key in range(4) if key != b], [0, 1, 2, 3]]


def generate_conditions():
    def spec(name, family="reference", layout="grouped", guard=None, restore=None,
             attention_noop=False, a=None, b=None):
        return {"id": name, "family": family, "layout": layout,
                "guard_keep_keys": guard, "restore": restore,
                "attention_noop": attention_noop, "a": a, "b": b,
                "is_correct_guard": guard == keep_keys(0, 3), "oracle": restore is not None}
    conditions = [spec("native", layout="native"), spec("grouped_canonical"),
                  spec("grouped_noop", attention_noop=True),
                  spec("restore_keys", family="oracle", restore="keys"),
                  spec("guard_restore_keys", family="oracle", guard=keep_keys(0, 3), restore="keys"),
                  spec("restore_values", family="oracle", restore="values")]
    conditions += [spec(f"guard_a{a}_b{b}", family="matched_guard", guard=keep_keys(a, b), a=a, b=b)
                   for a in (0, 2, 3) for b in (0, 1, 3)]
    return conditions


def main():
    path = Path("experiment4/conditions.json")
    if path.exists():
        raise SystemExit(f"Refusing to replace {path}")
    conditions = generate_conditions()
    path.write_text(json.dumps(conditions, indent=2) + "\n")
    print(json.dumps({"path": str(path), "conditions": len(conditions),
                      "matched_guards": 9, "correct_guard": "guard_a0_b3"}))


if __name__ == "__main__":
    main()
