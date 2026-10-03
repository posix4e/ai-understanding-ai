"""E4 local CPU interventions; the phase controller owns the public freeze.

Inputs are native interleaved tokens[N,9], targets[N], query_pair[N], values[N,4]
with value CLASS indices. The runner uses root-model hooks only. Guard patterns
are stable masked-softmax recomputations from the canonical-grouped run's own
layer-1 Q/K; no original activations are used by a non-oracle guard.

Actual layer-1 output diagnostics use (0,resid_post). Effective layer-2 input
diagnostics use (1,resid_pre), after any explicitly declared oracle transplant.
Every residual comparison is aligned by token identity to canonical grouped
order. Attention diagnostics retain all four heads separately.
"""

import math

import torch

from experiment4.conditions import GROUPED


METRICS = ("original_probability", "accuracy", "argmax_class", "class_probability",
           "l1_value_max_error_native", "l1_query_max_error_native", "l1_key_max_error_native",
           "l1_key_max_error_grouped", "l2_input_value_max_error_native", "l2_input_query_max_error_native",
           "l2_input_key_max_error_native", "l2_input_key_max_error_grouped",
           "l1_true_key_attention", "l2_correct_value_attention")


def validate_conditions(conditions):
    names = set()
    for condition in conditions:
        name = condition.get("id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Condition ids must be unique nonempty strings")
        names.add(name)
        if condition["layout"] not in ("native", "grouped") or condition["restore"] not in (None, "keys", "values"):
            raise ValueError(f"{name}: unsupported layout or restoration")
        guard = condition["guard_keep_keys"]
        if guard is not None:
            if len(guard) != 4:
                raise ValueError(f"{name}: guard must contain four value rows")
            for value, keys in enumerate(guard):
                if (len(keys) != value + 1 or len(set(keys)) != len(keys) or value not in keys
                        or any(key not in range(4) for key in keys)):
                    raise ValueError(f"{name}: invalid matched guard at value {value}")
        if guard is not None and condition["attention_noop"]:
            raise ValueError(f"{name}: guard and attention noop cannot be combined")
        if condition["layout"] == "native" and (guard is not None or condition["restore"] is not None or condition["attention_noop"]):
            raise ValueError("Native is an unmodified reference")


def _arrays(model, arrays):
    data = {key: torch.as_tensor(arrays[key], dtype=torch.long, device="cpu")
            for key in ("tokens", "targets", "query_pair", "values")}
    n = len(data["tokens"])
    if (data["tokens"].shape != (n, 9) or not n or data["targets"].shape != (n,)
            or data["query_pair"].shape != (n,) or data["values"].shape != (n, 4)):
        raise ValueError("Expected native tokens[N,9], targets[N], query_pair[N], values[N,4]")
    if (model.config.n_pairs, model.config.n_layers, model.config.n_heads) != (4, 2, 4):
        raise ValueError("Expected frozen four-pair/two-layer/four-head architecture")
    tokens, query, values = data["tokens"], data["query_pair"], data["values"]
    if torch.any((query < 0) | (query >= 4)) or torch.any((values < 0) | (values >= model.config.n_values)):
        raise ValueError("Invalid query/value class metadata")
    keys = tokens[:, :8:2]
    if torch.any((keys < 0) | (keys >= model.config.n_keys)):
        raise ValueError("Invalid key tokens")
    if not torch.equal(tokens[:, 1:8:2], values + model.config.n_keys):
        raise ValueError("values must be matching class indices, not token ids")
    rows = torch.arange(n)
    if not torch.equal(tokens[:, 8], keys[rows, query]) or not torch.equal(data["targets"], values[rows, query]):
        raise ValueError("Target/query metadata mismatch")
    for array in (keys, values):
        sorted_array = array.sort(1).values
        if torch.any(sorted_array[:, 1:] == sorted_array[:, :-1]):
            raise ValueError("Distinct keys and distinct values required")
    return data


def guarded_pattern(grouped_cache, keep):
    """Return grouped pattern with altered value rows only, stably normalized."""
    pattern = grouped_cache[(0, "attn_pattern")].clone()
    q, k = grouped_cache[(0, "q")], grouped_cache[(0, "k")]
    for value, keys in enumerate(keep):
        removed = [key for key in range(4) if key not in keys]
        if not removed:
            continue
        query_position = 4 + value
        scores = torch.einsum("bhd,bshd->bhs", q[:, query_position], k) / math.sqrt(q.shape[-1])
        scores[:, :, query_position + 1:] = -torch.inf
        scores[:, :, removed] = -torch.inf
        pattern[:, :, query_position, :] = scores.softmax(-1)
    if not torch.isfinite(pattern).all():
        raise ValueError("Nonfinite guarded attention pattern")
    return pattern


def _residual_differences(actual, native_grouped, grouped_original, prefix):
    return {prefix + "value_max_error_native": (actual[:, 4:8] - native_grouped[:, 4:8]).abs().amax((1, 2)),
            prefix + "query_max_error_native": (actual[:, 8] - native_grouped[:, 8]).abs().amax(1),
            prefix + "key_max_error_native": (actual[:, :4] - native_grouped[:, :4]).abs().amax((1, 2)),
            prefix + "key_max_error_grouped": (actual[:, :4] - grouped_original[:, :4]).abs().amax((1, 2))}


@torch.inference_mode()
def run(model, arrays, conditions, batch_size=128):
    if next(model.parameters()).device.type != "cpu" or batch_size < 1:
        raise ValueError("E4 requires a CPU model and positive batch_size")
    validate_conditions(conditions)
    data = _arrays(model, arrays)
    output = {condition["id"]: {metric: [] for metric in METRICS} for condition in conditions}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(data["tokens"]), batch_size):
            stop = start + batch_size
            tokens, targets = data["tokens"][start:stop], data["targets"][start:stop]
            query = data["query_pair"][start:stop]
            rows = torch.arange(len(tokens))
            native_logits, native_cache = model(tokens, return_cache=True)
            grouped_tokens = tokens[:, GROUPED]
            embedding = model.token_embedding(grouped_tokens) + model.position_embedding(torch.tensor(GROUPED))
            base_patch = {(0, "resid_pre"): (slice(None), embedding)}
            grouped_logits, grouped_cache = model(grouped_tokens, patch=base_patch, return_cache=True)
            native_residual = native_cache[(0, "resid_post")][:, GROUPED]
            grouped_residual = grouped_cache[(0, "resid_post")]
            for condition in conditions:
                native_layout = condition["layout"] == "native"
                if native_layout:
                    logits, cache = native_logits, native_cache
                else:
                    patch = dict(base_patch)
                    if condition["guard_keep_keys"] is not None:
                        pattern = guarded_pattern(grouped_cache, condition["guard_keep_keys"])
                        patch[(0, "attn_pattern")] = ([4, 5, 6, 7], pattern)
                    elif condition["attention_noop"]:
                        patch[(0, "attn_pattern")] = ([4, 5, 6, 7], grouped_cache[(0, "attn_pattern")])
                    if condition["restore"] is not None:
                        positions = [0, 1, 2, 3] if condition["restore"] == "keys" else [4, 5, 6, 7]
                        # Root hook modifies only these positions, leaving all
                        # unselected states endogenous to the current guard.
                        patch[(1, "resid_pre")] = (positions, native_residual)
                    if len(patch) == 1:
                        logits, cache = grouped_logits, grouped_cache
                    else:
                        logits, cache = model(grouped_tokens, patch=patch, return_cache=True)
                final_probability = logits[:, -1].softmax(-1)
                predicted = logits[:, -1].argmax(-1)
                l1_residual = cache[(0, "resid_post")]
                effective_input = cache[(1, "resid_pre")]
                if native_layout:
                    l1_residual = l1_residual[:, GROUPED]
                    effective_input = effective_input[:, GROUPED]
                    key_positions, value_positions = [0, 2, 4, 6], [1, 3, 5, 7]
                else:
                    key_positions, value_positions = [0, 1, 2, 3], [4, 5, 6, 7]
                values_at = torch.tensor(value_positions)
                metrics = {"original_probability": final_probability[rows, targets],
                           "accuracy": (predicted == targets).long(), "argmax_class": predicted,
                           "class_probability": final_probability,
                           **_residual_differences(l1_residual, native_residual, grouped_residual, "l1_"),
                           **_residual_differences(effective_input, native_residual, grouped_residual, "l2_input_"),
                           "l1_true_key_attention": cache[(0, "attn_pattern")][:, :, value_positions, key_positions].mean(-1),
                           "l2_correct_value_attention": cache[(1, "attn_pattern")][:, :, 8, :].gather(
                               2, values_at[query][:, None, None].expand(-1, 4, 1)).squeeze(-1)}
                for metric, tensor in metrics.items():
                    output[condition["id"]][metric].extend(tensor.tolist())
    finally:
        model.train(was_training)
    return output
