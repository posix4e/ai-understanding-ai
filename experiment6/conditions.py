"""Fifteen fixed GPT-2 visibility conditions; no model loading or evaluation.

CONDITIONS is JSON serializable. ``layers`` uses zero-based block indices;
``position_mode`` selects physical or original (canonical) token positions.
``grouped_noop`` explicitly supplies the complete physical causal mask at block
0. The three unpatched baselines have no layer masks.

The nine guards keep 1, 2, 3, and 4 key chunks respectively at V0..V3, always
including that value's own key. Parameter a is V1's extra retained key and b is
V2's excluded key. The correct guard is (a=0,b=3). Shams 0..7 enumerate the
remaining cells in lexicographic order over a=(0,2,3), b=(0,1,3).
"""
from copy import deepcopy

import torch

from experiment6.data import EncodedCase


def _spec(identifier, family, *, grouped=True, position_mode="canonical",
          layers=(), retained_keys=None, a=None, b=None):
    return {"id": identifier, "family": family, "grouped": grouped,
            "position_mode": position_mode, "layers": list(layers),
            "retained_keys": retained_keys, "a": a, "b": b}


def build_conditions():
    """Return independent JSON-safe definitions in the frozen execution order."""
    result = [
        _spec("native", "reference", grouped=False, position_mode="physical"),
        _spec("grouped_physical", "reference", position_mode="physical"),
        _spec("grouped_canonical", "reference"),
        _spec("grouped_noop", "noop", layers=(0,)),
    ]
    correct = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    result.append(_spec("guard_block0", "primary_guard", layers=(0,),
                        retained_keys=deepcopy(correct), a=0, b=3))
    sham = 0
    for a in (0, 2, 3):
        for b in (0, 1, 3):
            if (a, b) == (0, 3):
                continue
            keep = [[0], sorted([1, a]), [k for k in range(4) if k != b], list(range(4))]
            result.append(_spec(f"sham_{sham}", "matched_sham", layers=(0,),
                                retained_keys=keep, a=a, b=b))
            sham += 1
    result.extend([
        _spec("guard_block5", "secondary_guard", layers=(5,),
              retained_keys=deepcopy(correct), a=0, b=3),
        _spec("guard_all", "secondary_guard", layers=range(12),
              retained_keys=deepcopy(correct), a=0, b=3),
    ])
    return result


CONDITIONS = build_conditions()


def construct_mask(encoded: EncodedCase, condition: dict | str) -> torch.Tensor:
    """Return allowed edges [T,T] in the condition's physical token order.

Callers supply this mask to each block in ``condition['layers']``. Only rows
belonging to value chunks can be edited. No prefix, suffix, value, self, or own
key edge is removed, and no physically future edge is opened. Canonical IDs
are supplied independently via ``encoded.permutation(grouped=True)``.
"""
    if isinstance(condition, str):
        candidates = [item for item in CONDITIONS if item["id"] == condition]
        if not candidates:
            raise ValueError(f"unknown condition: {condition}")
        condition = candidates[0]
    if encoded.ids.ndim != 1 or len(encoded.chunks) != 8 or any(not c for c in encoded.chunks):
        raise ValueError("expected eight nonempty chunks and one-dimensional IDs")
    if any(len(encoded.chunks[2*i]) != 2 for i in range(4)):
        raise ValueError("each key chunk must contain exactly two tokens")
    permutation = encoded.permutation(grouped=condition["grouped"])
    length = encoded.ids.numel()
    inverse = torch.empty(length, dtype=torch.long)
    inverse[permutation] = torch.arange(length)
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    keep = condition["retained_keys"]
    if keep is None:
        return allowed
    if len(keep) != 4:
        raise ValueError("retained_keys must have one list per value")
    for i, keys in enumerate(keep):
        if (len(set(keys)) != len(keys) or i not in keys
                or any(type(k) is not int or k not in range(4) for k in keys)):
            raise ValueError("retained keys must be unique valid keys including own key")
        rows = inverse[encoded.chunks[2*i+1]]
        for j in range(4):
            if j not in keys:
                columns = inverse[encoded.chunks[2*j]]
                allowed[rows[:, None], columns[None, :]] = False
    return allowed
