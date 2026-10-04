"""Float64 GPT-2 diagnostic adapter with explicit causal or oracle masks.

Each layer mask is a COMPLETE allowed-edge Boolean matrix [T,T] or [B,T,T],
shared across heads. Layers without a mask retain physical causality. Future
physical edges require ``allow_future=True``; this is an oracle intervention,
not ordinary autoregressive inference. Selected layers temporarily bypass the
internal GPT-2 triangular bias so it cannot silently truncate an oracle mask.

ResidualPatch applies after the complete named block, before the next block's
LayerNorm. Captures include the patched state. Call without patches to obtain
clean donor states. All tensor rows refer to the current physical token order.
No padding, KV cache, dropout, concurrent calls, or model training is supported.
"""
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import GPT2LMHeadModel


@dataclass(frozen=True)
class ResidualPatch:
    positions: torch.Tensor  # CPU int64 [P], unique physical rows
    values: torch.Tensor  # CPU float64 [B,P,D]


@dataclass(frozen=True)
class AdapterOutput:
    logits: torch.Tensor  # final physical token, [B,vocabulary]
    post_block: dict[int, torch.Tensor]
    used_future_edges: bool

    @property
    def first_block(self):
        return self.post_block.get(0)


def checked_allowed(allowed, batch, length, allow_future):
    physical = torch.ones(length, length, dtype=torch.bool).tril()
    if not isinstance(allowed, torch.Tensor) or allowed.dtype != torch.bool:
        raise TypeError("complete masks must be Boolean tensors")
    if allowed.device.type != "cpu":
        raise ValueError("masks must be on CPU")
    if allowed.shape == (length, length):
        allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
    elif allowed.shape != (batch, length, length):
        raise ValueError("complete mask shape must be [T,T] or [B,T,T]")
    has_future = bool((allowed & ~physical).any())
    if has_future and not allow_future:
        raise ValueError("future physical edges require explicit oracle allow_future=True")
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("every attention row must retain at least one source")
    return allowed, has_future


def as_additive(allowed):
    return torch.zeros(allowed.shape, dtype=torch.float64).masked_fill(~allowed, -torch.inf).unsqueeze(1)


class GPT2DiagnosticAdapter:
    def __init__(self, model: GPT2LMHeadModel):
        if not isinstance(model, GPT2LMHeadModel):
            raise TypeError("expected GPT2LMHeadModel")
        if model.config.add_cross_attention:
            raise ValueError("cross attention is outside this adapter's contract")
        if model.config.reorder_and_upcast_attn:
            raise ValueError("reorder_and_upcast_attn would force float32 attention")
        torch.set_num_threads(4)
        self.model = model.to(device="cpu", dtype=torch.float64).eval()
        self.model.set_attn_implementation("eager")
        self.model.requires_grad_(False)
        self._active = False

    @classmethod
    def from_local(cls, path: str | Path = "work/models/gpt2"):
        """Load existing local float32 values, then promote exactly to float64."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        model = GPT2LMHeadModel.from_pretrained(
            str(path.resolve()), local_files_only=True, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager",
        )
        return cls(model)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                layer_masks: Mapping[int, torch.Tensor] | None = None,
                allow_future: bool = False,
                residual_patches: Mapping[int, ResidualPatch] | None = None,
                capture_layers: Sequence[int] | str | None = None) -> AdapterOutput:
        if self._active:
            raise RuntimeError("adapter calls must not overlap")
        if type(allow_future) is not bool:
            raise TypeError("allow_future must be an explicit Boolean")
        if (not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cpu"
                or input_ids.dtype != torch.long or input_ids.ndim != 2):
            raise ValueError("input_ids must be CPU int64 [B,T]")
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.model.config.n_positions:
            raise ValueError("empty input or context limit exceeded")
        if bool(((input_ids < 0) | (input_ids >= self.model.config.vocab_size)).any()):
            raise ValueError("input ID outside vocabulary")
        if position_ids is None:
            position_ids = torch.arange(length).expand(batch, -1)
        if (not isinstance(position_ids, torch.Tensor) or position_ids.device.type != "cpu"
                or position_ids.dtype != torch.long or position_ids.shape != input_ids.shape):
            raise ValueError("position_ids must be CPU int64 [B,T]")
        if bool(((position_ids < 0) | (position_ids >= self.model.config.n_positions)).any()):
            raise ValueError("position ID outside learned table")
        layer_count = len(self.model.transformer.h)

        def check_layer(layer):
            if type(layer) is not int or not 0 <= layer < layer_count:
                raise ValueError("invalid zero-based block index")

        masks, used_future = {}, False
        for layer, allowed in (layer_masks or {}).items():
            check_layer(layer)
            complete, has_future = checked_allowed(allowed, batch, length, allow_future)
            masks[layer] = as_additive(complete)
            used_future = used_future or has_future
        patches = dict(residual_patches or {})
        for layer, patch in patches.items():
            check_layer(layer)
            if not isinstance(patch, ResidualPatch):
                raise TypeError("residual_patches values must be ResidualPatch objects")
            rows, values = patch.positions, patch.values
            if (not isinstance(rows, torch.Tensor) or rows.device.type != "cpu"
                    or rows.dtype != torch.long or rows.ndim != 1 or rows.numel() < 1):
                raise ValueError("patch positions must be nonempty CPU int64 [P]")
            if (bool(((rows < 0) | (rows >= length)).any()) or len(rows.unique()) != len(rows)):
                raise ValueError("patch positions must be unique in-range rows")
            if (not isinstance(values, torch.Tensor) or values.device.type != "cpu"
                    or values.dtype != torch.float64
                    or values.shape != (batch, len(rows), self.model.config.n_embd)
                    or not bool(torch.isfinite(values).all())):
                raise ValueError("patch values must be finite CPU float64 [B,P,D]")
        if isinstance(capture_layers, str):
            if capture_layers != "all":
                raise ValueError("capture_layers string must be 'all'")
            capture = set(range(layer_count))
        else:
            capture = set(capture_layers or ())
        for layer in capture:
            check_layer(layer)
        base = as_additive(torch.ones(batch, length, length, dtype=torch.bool).tril())
        handles, biases, states = [], [], {}
        self._active = True
        try:
            for layer, mask in masks.items():
                module = self.model.transformer.h[layer].attn
                if not hasattr(module, "bias") or module.bias.dtype != torch.bool:
                    raise RuntimeError("unsupported GPT-2 internal causal buffer")
                biases.append((module, module.bias))
                module.bias = torch.ones_like(module.bias)
                signature = inspect.signature(module.forward)

                def replace_mask(module, args, kwargs, mask=mask, signature=signature):
                    bound = signature.bind_partial(*args, **kwargs)
                    bound.arguments["attention_mask"] = mask
                    return bound.args, bound.kwargs

                handles.append(module.register_forward_pre_hook(replace_mask, with_kwargs=True))
            for layer in sorted(capture | set(patches)):
                def patch_and_capture(module, args, output, layer=layer):
                    residual = output[0]
                    if layer in patches:
                        patch = patches[layer]
                        residual = residual.clone()
                        residual[:, patch.positions, :] = patch.values
                    if layer in capture:
                        states[layer] = residual.detach().clone()
                    return (residual,) + output[1:]
                handles.append(self.model.transformer.h[layer].register_forward_hook(patch_and_capture))
            output = self.model.transformer(
                input_ids=input_ids, position_ids=position_ids, attention_mask=base,
                use_cache=False, return_dict=True, output_attentions=False,
                output_hidden_states=False,
            )
            logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
            if logits.dtype != torch.float64:
                raise RuntimeError("unexpected inference dtype")
            return AdapterOutput(logits, states, used_future)
        finally:
            for handle in reversed(handles):
                handle.remove()
            for module, original in reversed(biases):
                module.bias = original
            self._active = False

    __call__ = forward
