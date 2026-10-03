"""CPU-only layout-sharded E5 evaluation using the unchanged root-model hooks.

run_layout(model, arrays, layout, conditions=None, batch_size=128) returns
{local_condition_id: {metric: numpy_array}}. Arrays are float32 except int64
argmax_class/int8 accuracy; all have shape [N], except class_probability[N,16].
No data/checkpoint paths are accessed. The phase controller enforces freezing.
Input tokens[N,9] are NATIVE interleaved, with targets/query_pair[N] and value
CLASS indices values[N,4]. Each layout fixes canonical position IDs.

All non-oracle interventions use current-layout Q/K only. Native caches supply
diagnostics, the native own-key-self reference, and ONLY the explicit oracle's
second-layer key transplant. Own-key-self identity concerns first-layer value
states, not final outputs. First-layer actual output and second-layer effective
input are deliberately measured separately.
"""

import math

import numpy as np
import torch


METRICS = ("original_probability", "accuracy", "argmax_class", "class_probability",
           "l1_value_max_error_native", "l1_query_max_error_native", "l1_key_max_error_native",
           "l1_key_max_error_baseline", "l1_value_max_error_own_key_self",
           "l2_input_value_max_error_native", "l2_input_query_max_error_native",
           "l2_input_key_max_error_native", "class_probability_max_error_native")


def validate_layout(layout, conditions):
    indices = layout["layout_indices"]
    if sorted(indices) != list(range(9)) or indices[-1] != 8:
        raise ValueError("Layout must permute eight dictionary identities and retain query last")
    if any(indices.index(2*i) > indices.index(2*i+1) for i in range(4)):
        raise ValueError("Every key must precede its paired value")
    names = set()
    for c in conditions:
        if not isinstance(c.get("id"), str) or not c["id"] or c["id"] in names:
            raise ValueError("Condition names must be unique nonempty strings within a layout")
        names.add(c["id"])
        if c["kind"] not in ("unguarded", "noop", "keyguard", "logical_prefix", "own_key_self"):
            raise ValueError("Unsupported condition kind")
        if c.get("oracle") and (c["kind"] != "keyguard" or c.get("keep_keys") != [list(range(i+1)) for i in range(4)]
                                 or [t for t in indices if t % 2 and t < 8] != [1, 3, 5, 7]):
            raise ValueError("Oracle is defined only for the correct guard in the primary family")
        if c["kind"] == "keyguard":
            keep = c["keep_keys"]
            if len(keep) != 4:
                raise ValueError("Expected four key-retention rows")
            for i, row in enumerate(keep):
                if i not in row or len(set(row)) != len(row) or any(j not in range(4) or indices.index(2*j) > indices.index(2*i+1) for j in row):
                    raise ValueError("Retained keys must be unique, available, and include the own key")
        elif c.get("keep_keys") is not None:
            raise ValueError("Only keyguard accepts keep_keys")
    if not conditions:
        raise ValueError("At least one condition is required")


def _arrays(model, arrays):
    if (model.config.n_pairs, model.config.n_layers, model.config.n_heads, model.config.n_values) != (4, 2, 4, 16):
        raise ValueError("Expected the fixed E5 model architecture")
    data = {name: torch.as_tensor(arrays[name], dtype=torch.long, device="cpu")
            for name in ("tokens", "targets", "query_pair", "values")}
    tokens, targets, query, values = (data[k] for k in ("tokens", "targets", "query_pair", "values"))
    n = len(tokens)
    if not n or tokens.shape != (n, 9) or targets.shape != (n,) or query.shape != (n,) or values.shape != (n, 4):
        raise ValueError("Expected native tokens[N,9], targets[N], query_pair[N], values[N,4]")
    keys = tokens[:, :8:2]
    if torch.any((keys < 0) | (keys >= model.config.n_keys)) or torch.any((values < 0) | (values >= 16)) or torch.any((query < 0) | (query >= 4)):
        raise ValueError("Invalid key, query, or value class index")
    rows = torch.arange(n)
    if not torch.equal(tokens[:, 1:8:2], values + model.config.n_keys) or not torch.equal(tokens[:, 8], keys[rows, query]) or not torch.equal(targets, values[rows, query]):
        raise ValueError("Native tokens and target/query/value metadata disagree")
    if any(torch.any(x.sort(1).values[:, 1:] == x.sort(1).values[:, :-1]) for x in (keys, values)):
        raise ValueError("Dictionary keys and values must each be distinct")
    return data


def allowed_mask(layout, condition):
    """Boolean [9,9] physical attention support, retaining ordinary causal rows."""
    ids = layout["layout_indices"]
    allowed = torch.ones(9, 9, dtype=torch.bool).tril()
    kind = condition["kind"]
    if kind in ("unguarded", "noop"):
        return allowed
    for i in range(4):
        p = ids.index(2*i+1)
        for source, identity in enumerate(ids):
            if kind == "keyguard":
                remove = identity < 8 and identity % 2 == 0 and identity//2 not in condition["keep_keys"][i]
            elif kind == "logical_prefix":
                remove = identity > 2*i+1
            elif kind == "own_key_self":
                remove = identity not in (2*i, 2*i+1)
            else:
                raise ValueError("Unknown mask kind")
            if remove:
                allowed[p, source] = False
    return allowed


def masked_pattern(cache, layout, condition):
    """Stable direct masked-softmax; unchanged rows remain bitwise unchanged."""
    original = cache[(0, "attn_pattern")]
    if condition["kind"] in ("noop", "unguarded"):
        return original
    pattern = original.clone()
    mask = allowed_mask(layout, condition)
    causal = torch.ones(9, 9, dtype=torch.bool).tril()
    selected = torch.where(torch.any(mask != causal, dim=1))[0]
    if len(selected):
        q, k = cache[(0, "q")], cache[(0, "k")]
        scores = torch.einsum("bthd,bshd->bhts", q[:, selected], k) / math.sqrt(q.shape[-1])
        scores.masked_fill_(~mask[selected][None, None], -torch.inf)
        pattern[:, :, selected, :] = scores.softmax(-1)
    if not torch.isfinite(pattern).all():
        raise ValueError("Nonfinite masked attention")
    return pattern


def _max_rows(actual, reference, positions):
    return (actual[:, positions] - reference[:, positions]).abs().amax((1, 2))


@torch.inference_mode()
def run_layout(model, arrays, layout, conditions=None, batch_size=128):
    if next(model.parameters()).device.type != "cpu" or batch_size < 1:
        raise ValueError("CPU model and positive batch size required")
    conditions = layout["conditions"] if conditions is None else conditions
    validate_layout(layout, conditions)
    data = _arrays(model, arrays)
    n = len(data["tokens"])
    output = {c["id"]: {metric: np.empty((n, 16) if metric == "class_probability" else n,
                                        dtype=np.int64 if metric == "argmax_class" else np.int8 if metric == "accuracy" else np.float32)
                             for metric in METRICS} for c in conditions}
    ids = layout["layout_indices"]
    key_positions = [ids.index(2*i) for i in range(4)]
    value_positions = [ids.index(2*i+1) for i in range(4)]
    native_layout = {"layout_indices": list(range(9))}
    own_condition = {"kind": "own_key_self"}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            native_tokens = data["tokens"][start:stop]
            targets = data["targets"][start:stop]
            native_logits, native_cache = model(native_tokens, return_cache=True)
            native_probability = native_logits[:, -1].softmax(-1)
            native_residual = native_cache[(0, "resid_post")][:, ids]
            own_pattern = masked_pattern(native_cache, native_layout, own_condition)
            _, own_cache = model(native_tokens, patch={(0, "attn_pattern"): ([1, 3, 5, 7], own_pattern)}, return_cache=True)
            own_residual = own_cache[(0, "resid_post")][:, ids]
            tokens = native_tokens[:, ids]
            embedding = model.token_embedding(tokens) + model.position_embedding(torch.tensor(ids))
            base_patch = {(0, "resid_pre"): (slice(None), embedding)}
            if ids == list(range(9)):
                base_logits, base_cache = native_logits, native_cache
            else:
                base_logits, base_cache = model(tokens, patch=base_patch, return_cache=True)
            baseline_residual = base_cache[(0, "resid_post")]
            rows = torch.arange(stop-start)
            for c in conditions:
                if c["kind"] == "unguarded":
                    logits, cache = base_logits, base_cache
                else:
                    patch = dict(base_patch)
                    patch[(0, "attn_pattern")] = (value_positions, masked_pattern(base_cache, layout, c))
                    if c.get("oracle"):
                        patch[(1, "resid_pre")] = (key_positions, native_residual)
                    logits, cache = model(tokens, patch=patch, return_cache=True)
                probability = logits[:, -1].softmax(-1)
                predicted = logits[:, -1].argmax(-1)
                l1 = cache[(0, "resid_post")]
                l2 = cache[(1, "resid_pre")]
                metrics = {"original_probability": probability[rows, targets], "accuracy": (predicted == targets),
                           "argmax_class": predicted, "class_probability": probability,
                           "l1_value_max_error_native": _max_rows(l1, native_residual, value_positions),
                           "l1_query_max_error_native": _max_rows(l1, native_residual, [8]),
                           "l1_key_max_error_native": _max_rows(l1, native_residual, key_positions),
                           "l1_key_max_error_baseline": _max_rows(l1, baseline_residual, key_positions),
                           "l1_value_max_error_own_key_self": _max_rows(l1, own_residual, value_positions),
                           "l2_input_value_max_error_native": _max_rows(l2, native_residual, value_positions),
                           "l2_input_query_max_error_native": _max_rows(l2, native_residual, [8]),
                           "l2_input_key_max_error_native": _max_rows(l2, native_residual, key_positions),
                           "class_probability_max_error_native": (probability-native_probability).abs().amax(1)}
                for metric, tensor in metrics.items():
                    output[c["id"]][metric][start:stop] = tensor.cpu().numpy()
    finally:
        model.train(was_training)
    return output
