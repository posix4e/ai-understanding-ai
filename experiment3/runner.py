"""E3 CPU forwards for already-frozen conditions; caller enforces phase lock.

run(model, arrays, conditions, batch_size=128) accepts original interleaved
tokens[N,9], targets[N] (class indices), query_pair[N], and values[N,4]
(class indices, not token ids). Each condition returns per-case probability,
accuracy and argmax arrays plus four attention diagnostics of shape [N,4].
Attention head indices are never selected or weighted by observed behavior.
L1 diagnostics average over all four values; L2 diagnostics use final query.
Grouped-native has no defined slot map: its slot metrics intentionally equal
original metrics, and consumers must retain slot_mapping_defined=false.
"""

import torch


METRICS = ("original_probability", "slot_probability", "original_accuracy", "slot_accuracy",
           "argmax_class", "class_probability", "l1_true_key_attention", "l1_slot_key_attention",
           "l2_original_value_attention", "l2_slot_value_attention")


def validate_conditions(conditions):
    names = set()
    for condition in conditions:
        name = condition.get("id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Condition ids must be unique nonempty strings")
        names.add(name)
        layout = condition["layout_indices"]
        if sorted(layout) != list(range(9)) or layout[-1] != 8:
            raise ValueError(f"{name}: layout must permute dictionary tokens and retain query last")
        positions = condition["position_ids"]
        if positions is not None and (sorted(positions) != list(range(9)) or positions[-1] != 8):
            raise ValueError(f"{name}: position ids must permute 0..8 and retain query id 8")
        for field in ("permutation", "slot_value_by_query", "slot_key_by_value"):
            if sorted(condition[field]) != list(range(4)):
                raise ValueError(f"{name}: {field} must be a four-slot permutation")
        key_positions = [layout.index(2 * i) for i in range(4)]
        value_positions = [layout.index(2 * i + 1) for i in range(4)]
        if any(key_positions[i] >= value_positions[i] for i in range(4)):
            raise ValueError(f"{name}: true key must precede its value")
        if any(key_positions[condition['slot_key_by_value'][i]] >= value_positions[i] for i in range(4)):
            raise ValueError(f"{name}: slot-associated key must be causally available")


def _validate_arrays(model, arrays):
    data = {name: torch.as_tensor(arrays[name], dtype=torch.long, device="cpu")
            for name in ("tokens", "targets", "query_pair", "values")}
    tokens, targets, query_pair, values = (data[name] for name in ("tokens", "targets", "query_pair", "values"))
    n = len(tokens)
    if tokens.shape != (n, 9) or n == 0 or targets.shape != (n,) or query_pair.shape != (n,) or values.shape != (n, 4):
        raise ValueError("Expected tokens[N,9], targets[N], query_pair[N], values[N,4] with N>0")
    if model.config.n_pairs != 4 or model.config.n_layers != 2 or model.config.n_heads != 4:
        raise ValueError("E3 requires the frozen four-pair/two-layer/four-head architecture")
    if torch.any((query_pair < 0) | (query_pair >= 4)):
        raise ValueError("Invalid query_pair")
    keys = tokens[:, :8:2]
    if torch.any((keys < 0) | (keys >= model.config.n_keys)):
        raise ValueError("Invalid key tokens")
    if torch.any((values < 0) | (values >= model.config.n_values)):
        raise ValueError("values must be value-class indices")
    if not torch.equal(values + model.config.n_keys, tokens[:, 1:8:2]):
        raise ValueError("Value metadata does not match tokens")
    rows = torch.arange(n)
    if not torch.equal(targets, values[rows, query_pair]) or not torch.equal(tokens[:, 8], keys[rows, query_pair]):
        raise ValueError("Query/target metadata does not match tokens")
    if torch.any(keys.sort(1).values[:, 1:] == keys.sort(1).values[:, :-1]) or torch.any(values.sort(1).values[:, 1:] == values.sort(1).values[:, :-1]):
        raise ValueError("Each dictionary must contain distinct keys and distinct values")
    return data


@torch.inference_mode()
def run(model, arrays, conditions, batch_size=128):
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("E3 requires local CPU execution")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validate_conditions(conditions)
    data = _validate_arrays(model, arrays)
    output = {condition["id"]: {metric: [] for metric in METRICS} for condition in conditions}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(data["tokens"]), batch_size):
            stop = start + batch_size
            original = data["tokens"][start:stop]
            query_pair = data["query_pair"][start:stop]
            targets = data["targets"][start:stop]
            values = data["values"][start:stop]
            rows = torch.arange(len(original))
            for condition in conditions:
                layout = condition["layout_indices"]
                tokens = original[:, layout]
                patch = None
                if condition["position_ids"] is not None:
                    position_ids = torch.tensor(condition["position_ids"], dtype=torch.long)
                    residual = model.token_embedding(tokens) + model.position_embedding(position_ids)
                    patch = {(0, "resid_pre"): (slice(None), residual)}
                logits, cache = model(tokens, patch=patch, return_cache=True)
                probability = logits[:, -1].softmax(-1)
                predicted = logits[:, -1].argmax(-1)
                slot_pairs = torch.tensor(condition["slot_value_by_query"])[query_pair]
                slot_targets = values[rows, slot_pairs]
                key_positions = torch.tensor([layout.index(2 * i) for i in range(4)])
                value_positions = torch.tensor([layout.index(2 * i + 1) for i in range(4)])
                slot_key_positions = key_positions[torch.tensor(condition["slot_key_by_value"])]
                layer1 = cache[(0, "attn_pattern")]
                layer2_query = cache[(1, "attn_pattern")][:, :, 8, :]
                original_index = value_positions[query_pair][:, None, None].expand(-1, 4, 1)
                slot_index = value_positions[slot_pairs][:, None, None].expand(-1, 4, 1)
                metrics = {"original_probability": probability[rows, targets],
                           "slot_probability": probability[rows, slot_targets],
                           "original_accuracy": (predicted == targets).long(),
                           "slot_accuracy": (predicted == slot_targets).long(),
                           "argmax_class": predicted,
                           "class_probability": probability,
                           "l1_true_key_attention": layer1[:, :, value_positions, key_positions].mean(-1),
                           "l1_slot_key_attention": layer1[:, :, value_positions, slot_key_positions].mean(-1),
                           "l2_original_value_attention": layer2_query.gather(2, original_index).squeeze(-1),
                           "l2_slot_value_attention": layer2_query.gather(2, slot_index).squeeze(-1)}
                for metric, tensor in metrics.items():
                    output[condition["id"]][metric].extend(tensor.tolist())
    finally:
        model.train(was_training)
    return output
