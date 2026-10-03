"""Local CPU intervention runner for Experiment 2.

Layers and heads are zero-indexed. A condition has an ``id`` and ``ops`` list.
Each operation specifies layer, component (z or mlp_out), positions, kind,
and (for z) heads. kind is zero, mean, resample, noop, or rescue. Rescue heads
are the heads restored from the CLEAN recipient; all other heads receive the
specified corruption (zero, mean, or resample). Resample donor names refer to
arrays['donor_tokens'] or arrays['matched_donor_tokens'] and default to the
former. Calibration means are computed separately and never updated by run().

At most one operation per (layer,component) is allowed within a condition.
Cross-layer and z+MLP operations run in causal order. Downstream unselected
heads/components remain endogenous to upstream interventions. Clean donor
activations, resampled donor activations, and calibration means are fixed.
"""

import torch


COMPONENT_ORDER = {"z": 0, "mlp_out": 1}
KINDS = {"zero", "mean", "resample", "noop", "rescue"}


def _tokens(value):
    return torch.as_tensor(value, dtype=torch.long, device="cpu")


def _require_cpu(model):
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Experiment 2 runner requires a CPU model")


@torch.inference_mode()
def calibrate(model, calibration_tokens, batch_size=128):
    """Return per-position means [1,T,...] from independent calibration cases.

    Output keys are (zero-indexed layer, component) for z and mlp_out. The
    caller is responsible for fixing the calibration inputs before discovery
    and saving/hashing these tensors before confirmation.
    """
    _require_cpu(model)
    tokens = _tokens(calibration_tokens)
    if tokens.ndim != 2 or len(tokens) == 0 or batch_size < 1:
        raise ValueError("calibration_tokens must be a nonempty [B,T] array")
    sums = {}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(tokens), batch_size):
            _, cache = model(tokens[start:start + batch_size], return_cache=True)
            for key, activation in cache.items():
                if key[1] in COMPONENT_ORDER:
                    part = activation.to(torch.float64).sum(dim=0, keepdim=True)
                    sums[key] = sums.get(key, 0) + part
    finally:
        model.train(was_training)
    return {key: (value / len(tokens)).to(torch.float32) for key, value in sums.items()}


def validate_conditions(conditions, config):
    """Validate JSON condition specs without executing a model forward."""
    ids = set()
    for condition in conditions:
        name = condition.get("id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Condition ids must be unique nonempty strings")
        ids.add(name)
        if not isinstance(condition.get("ops"), list):
            raise ValueError(f"{name}: ops must be a list")
        sites = set()
        for op in condition["ops"]:
            layer, component = op.get("layer"), op.get("component", "z")
            if not isinstance(layer, int) or not 0 <= layer < config.n_layers:
                raise ValueError(f"{name}: invalid layer {layer}")
            if component not in COMPONENT_ORDER or (layer, component) in sites:
                raise ValueError(f"{name}: invalid or repeated site {(layer, component)}")
            sites.add((layer, component))
            positions = op.get("positions")
            if (not isinstance(positions, list) or not positions
                    or len(set(positions)) != len(positions)
                    or any(not isinstance(p, int) or not 0 <= p < config.seq_len for p in positions)):
                raise ValueError(f"{name}: positions must be unique valid indices")
            kind = op.get("kind")
            if kind not in KINDS:
                raise ValueError(f"{name}: invalid kind {kind}")
            if component == "z":
                heads = op.get("heads")
                if (not isinstance(heads, list) or len(set(heads)) != len(heads)
                        or any(not isinstance(h, int) or not 0 <= h < config.n_heads for h in heads)):
                    raise ValueError(f"{name}: heads must be unique valid indices")
                if not heads and kind != "rescue":
                    raise ValueError(f"{name}: empty heads are only allowed for rescue")
            elif "heads" in op or kind == "rescue":
                raise ValueError(f"{name}: MLP operations cannot have heads or use rescue")
            if kind == "rescue" and op.get("corruption") not in {"zero", "mean", "resample"}:
                raise ValueError(f"{name}: rescue requires explicit corruption")
            if kind == "resample" or (kind == "rescue" and op["corruption"] == "resample"):
                if op.get("donor", "donor_tokens") not in {"donor_tokens", "matched_donor_tokens"}:
                    raise ValueError(f"{name}: unsupported donor array")


def _replacement(op, current, key, means, donor_caches):
    kind = op["corruption"] if op["kind"] == "rescue" else op["kind"]
    if kind == "zero":
        return torch.zeros_like(current)
    if kind == "mean":
        if means is None or key not in means:
            raise ValueError(f"Missing frozen calibration mean for {key}")
        mean = torch.as_tensor(means[key], dtype=current.dtype, device="cpu")
        expected = (1, *current.shape[1:])
        if tuple(mean.shape) != expected:
            raise ValueError(f"Mean {key}: expected {expected}, got {tuple(mean.shape)}")
        return mean.expand_as(current)
    if kind == "resample":
        return donor_caches[op.get("donor", "donor_tokens")][key]
    if kind == "noop":
        return current
    raise ValueError(f"Unsupported corruption {kind}")


def _metrics(logits, targets):
    final = logits[:, -1]
    rows = torch.arange(len(targets))
    target_logit = final[rows, targets]
    others = final.clone()
    others[rows, targets] = -torch.inf
    return {"target_probability": final.softmax(-1)[rows, targets].tolist(),
            "accuracy": (final.argmax(-1) == targets).to(torch.int64).tolist(),
            "logit_margin": (target_logit - others.max(-1).values).tolist()}


@torch.inference_mode()
def run(model, arrays, conditions, means=None, batch_size=128):
    """Return {condition_id: {metric: per_case_list}} for frozen conditions.

    arrays requires tokens [B,T] and targets [B]. Optional donor_tokens and
    matched_donor_tokens must have the same shape as tokens. run() does not
    create, modify, choose, or inspect a protocol/data split; the caller must
    enforce the discovery/confirmation boundary before invoking it.
    """
    _require_cpu(model)
    validate_conditions(conditions, model.config)
    tokens, targets = _tokens(arrays["tokens"]), _tokens(arrays["targets"])
    if tokens.ndim != 2 or tokens.shape[1] != model.config.seq_len or len(tokens) == 0:
        raise ValueError("tokens must be nonempty [B,seq_len]")
    if targets.shape != (len(tokens),) or torch.any((targets < 0) | (targets >= model.config.n_values)):
        raise ValueError("targets must be valid value classes [B]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    donor_names = {op.get("donor", "donor_tokens")
                   for c in conditions for op in c["ops"]
                   if op["kind"] == "resample" or
                   (op["kind"] == "rescue" and op["corruption"] == "resample")}
    donors = {}
    for name in donor_names:
        if name not in arrays:
            raise ValueError(f"Missing required donor array {name}")
        donors[name] = _tokens(arrays[name])
        if donors[name].shape != tokens.shape:
            raise ValueError(f"{name} must have the same shape as tokens")
    outputs = {c["id"]: {metric: [] for metric in
                          ("target_probability", "accuracy", "logit_margin")}
               for c in conditions}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(tokens), batch_size):
            stop = start + batch_size
            batch = tokens[start:stop]
            clean_logits, clean_cache = model(batch, return_cache=True)
            donor_caches = {name: model(value[start:stop], return_cache=True)[1]
                            for name, value in donors.items()}
            for condition in conditions:
                patch, current_cache, logits = {}, clean_cache, clean_logits
                ops = sorted(condition["ops"], key=lambda op:
                             (op["layer"], COMPONENT_ORDER[op.get("component", "z")]))
                for index, op in enumerate(ops):
                    key = (op["layer"], op.get("component", "z"))
                    current = current_cache[key]
                    replacement = _replacement(op, current, key, means, donor_caches)
                    patched = current.clone()
                    for position in op["positions"]:
                        if key[1] == "z":
                            if op["kind"] == "rescue":
                                patched[:, position] = replacement[:, position]
                                patched[:, position, op["heads"]] = clean_cache[key][:, position, op["heads"]]
                            else:
                                patched[:, position, op["heads"]] = replacement[:, position, op["heads"]]
                        else:
                            patched[:, position] = replacement[:, position]
                    patch[key] = (op["positions"], patched)
                    if index + 1 < len(ops):
                        logits, current_cache = model(batch, patch=patch, return_cache=True)
                    else:
                        logits = model(batch, patch=patch)
                result = _metrics(logits, targets[start:stop])
                for metric, values in result.items():
                    outputs[condition["id"]][metric].extend(values)
    finally:
        model.train(was_training)
    return outputs
