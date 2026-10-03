"""Offline GPT-2 inference with explicit positions and layer-specific deletions.

``layer_masks[layer]`` is an allowed-edge bool tensor [T,T] or [B,T,T].
The adapter intersects it with physical causality; True never opens a future
edge. Every row must retain at least one physically available predecessor.
Masks apply to all heads of exactly the specified zero-based layer. Calls are
sequential: temporary hooks mean concurrent calls on this model are unsupported.

Inputs have no padding and share a length. Position IDs identify the original
token coordinates, not physical causal order. Returned logits are only for the
last physical token; first_block is the residual after the complete first GPT-2
block (attention, residual, MLP, residual), before the next block's LayerNorm.
"""
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Mapping

import torch
from transformers import GPT2LMHeadModel


@dataclass(frozen=True)
class AdapterOutput:
    logits: torch.Tensor
    first_block: torch.Tensor | None = None


def causal_allowed(allowed: torch.Tensor | None, batch: int, length: int) -> torch.Tensor:
    """Validate, broadcast, and intersect a requested mask with causal order."""
    physical = torch.ones(length, length, dtype=torch.bool).tril()
    if allowed is None:
        result = physical.unsqueeze(0).expand(batch, -1, -1)
    else:
        if not isinstance(allowed, torch.Tensor) or allowed.dtype != torch.bool:
            raise TypeError("allowed attention masks must be boolean tensors")
        if allowed.device.type != "cpu":
            raise ValueError("attention masks must be on CPU")
        if allowed.shape == (length, length):
            allowed = allowed.unsqueeze(0).expand(batch, -1, -1)
        elif allowed.shape != (batch, length, length):
            raise ValueError("mask shape must be [T,T] or [B,T,T]")
        result = allowed & physical
    if not bool(result.any(dim=-1).all()):
        raise ValueError("each attention row must retain a physical predecessor")
    return result


def additive_mask(allowed: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,T] to eager attention's float32 [B,1,T,T] mask."""
    return torch.zeros(allowed.shape, dtype=torch.float32).masked_fill(~allowed, -torch.inf).unsqueeze(1)


class GPT2Adapter:
    def __init__(self, model: GPT2LMHeadModel):
        if not isinstance(model, GPT2LMHeadModel):
            raise TypeError("expected GPT2LMHeadModel")
        if model.config.add_cross_attention:
            raise ValueError("cross-attention models are outside this experiment")
        torch.set_num_threads(4)
        self.model = model.to(device="cpu", dtype=torch.float32).eval()
        self.model.set_attn_implementation("eager")
        self.model.requires_grad_(False)
        self._active = False

    @classmethod
    def from_local(cls, path: str | Path = "work/models/gpt2") -> "GPT2Adapter":
        """Load existing local safetensors only; never contact the model hub."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        model = GPT2LMHeadModel.from_pretrained(
            str(path.resolve()), local_files_only=True, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager",
        )
        return cls(model)

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
        layer_masks: Mapping[int, torch.Tensor] | None = None,
        capture_first_block: bool = False,
    ) -> AdapterOutput:
        if self._active:
            raise RuntimeError("adapter calls must not overlap")
        if input_ids.device.type != "cpu" or input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be CPU int64 [B,T]")
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.model.config.n_positions:
            raise ValueError("empty batch or sequence exceeds model context")
        if bool(((input_ids < 0) | (input_ids >= self.model.config.vocab_size)).any()):
            raise ValueError("input token outside vocabulary")
        if position_ids is None:
            position_ids = torch.arange(length).expand(batch, -1)
        if (position_ids.device.type != "cpu" or position_ids.dtype != torch.long
                or position_ids.shape != input_ids.shape):
            raise ValueError("position_ids must be CPU int64 [B,T]")
        if bool(((position_ids < 0) | (position_ids >= self.model.config.n_positions)).any()):
            raise ValueError("position outside learned embedding table")
        masks = {}
        for layer, allowed in (layer_masks or {}).items():
            if type(layer) is not int or not 0 <= layer < len(self.model.transformer.h):
                raise ValueError("invalid zero-based layer index")
            masks[layer] = additive_mask(causal_allowed(allowed, batch, length))
        base = additive_mask(causal_allowed(None, batch, length))
        handles = []
        captured = []
        self._active = True
        try:
            for layer, mask in masks.items():
                module = self.model.transformer.h[layer].attn
                signature = inspect.signature(module.forward)

                def replace_mask(module, args, kwargs, mask=mask, signature=signature):
                    bound = signature.bind_partial(*args, **kwargs)
                    bound.arguments["attention_mask"] = mask
                    return bound.args, bound.kwargs

                handles.append(module.register_forward_pre_hook(replace_mask, with_kwargs=True))
            if capture_first_block:
                def capture(module, args, output):
                    captured.append(output[0].detach().clone())
                handles.append(self.model.transformer.h[0].register_forward_hook(capture))
            # Explicit 4D physical mask also avoids interpreting nonmonotonic
            # canonical position IDs as packed-sequence boundaries.
            output = self.model.transformer(
                input_ids=input_ids, position_ids=position_ids,
                attention_mask=base, use_cache=False, return_dict=True,
                output_attentions=False, output_hidden_states=False,
            )
            logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
            return AdapterOutput(logits=logits, first_block=captured[0] if captured else None)
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    __call__ = forward
