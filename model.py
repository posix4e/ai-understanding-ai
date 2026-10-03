"""Small causal transformer and reproducible random dictionary task.

Forward returns logits [batch, position, n_values]. Only the final position is
trained. With return_cache=True it returns (logits, cache). Cache keys are
(layer_number, component_name). Residual/branch activations have shape [B,T,D];
q/k/v have shape [B,T,H,D/H]; attn_pattern has shape [B,H,T,T].

Intervention contract: patch[(layer, component)] = (positions, donor_tensor).
The donor is a full tensor with exactly the corresponding cache shape. int,
list[int], tensor indices, and slices are accepted as positions. The selected
positions are copied for every batch item. For attn_pattern, positions index
the query-position axis (axis 2); all other components use axis 1. Patches run
before caching and before downstream computation. Patch q/k after the
normalized residual is projected, and patch attn_out after the output
projection. resid_pre patches do not retroactively change the previous layer's
cache. To intervene on one head, clone a donor/recipient cache tensor and edit
the desired head before supplying the full tensor.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class ModelConfig:
    n_keys: int = 16
    n_values: int = 16
    n_pairs: int = 4
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 128
    n_layers: int = 2
    embedding_std: float = 0.02

    @property
    def seq_len(self):
        return 2 * self.n_pairs + 1

    def to_dict(self):
        return asdict(self)


def generate_batch(batch_size, generator, config=None):
    """Generate independent dictionaries using the supplied CPU torch.Generator.

    Keys and values are independently sampled without replacement. Keys are
    token ids [0,n_keys), value tokens [n_keys,n_keys+n_values). Targets are
    value CLASS indices [0,n_values). metadata['values'] uses class indices.
    The final token is a uniformly chosen key already in the dictionary.
    """
    config = config or ModelConfig()
    if config.n_pairs > min(config.n_keys, config.n_values):
        raise ValueError("n_pairs exceeds available distinct keys/values")
    keys = torch.rand(batch_size, config.n_keys, generator=generator).argsort(1)
    values = torch.rand(batch_size, config.n_values, generator=generator).argsort(1)
    keys, values = keys[:, :config.n_pairs], values[:, :config.n_pairs]
    query_pair = torch.randint(config.n_pairs, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    tokens = torch.empty(batch_size, config.seq_len, dtype=torch.long)
    tokens[:, :-1:2] = keys
    tokens[:, 1::2] = values + config.n_keys
    tokens[:, -1] = keys[rows, query_pair]
    targets = values[rows, query_pair]
    metadata = {"query_pair": query_pair, "keys": keys, "values": values}
    return tokens, targets, metadata


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.attn_projection = nn.Linear(config.d_model, config.d_model)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_mlp),
                                 nn.GELU(), nn.Linear(config.d_mlp, config.d_model))

    def forward(self, residual, layer, patch, cache):
        def activation(name, tensor):
            key = (layer, name)
            if key in patch:
                positions, donor = patch[key]
                if tuple(donor.shape) != tuple(tensor.shape):
                    raise ValueError(f"{key}: donor {donor.shape} != recipient {tensor.shape}")
                tensor = tensor.clone()
                axis = 2 if name == "attn_pattern" else 1
                index = [slice(None)] * tensor.ndim
                index[axis] = positions
                index = tuple(index)
                tensor[index] = donor.to(device=tensor.device, dtype=tensor.dtype)[index]
            if cache is not None:
                cache[key] = tensor.detach().clone()
            return tensor

        residual = activation("resid_pre", residual)
        batch, length, width = residual.shape
        heads = self.config.n_heads
        head_dim = width // heads
        projected = self.qkv(self.ln_attn(residual)).reshape(batch, length, 3, heads, head_dim)
        q = activation("q", projected[:, :, 0])
        k = activation("k", projected[:, :, 1])
        v = activation("v", projected[:, :, 2])
        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
        causal_mask = torch.ones(length, length, device=residual.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        pattern = activation("attn_pattern", scores.softmax(dim=-1))
        attention = torch.einsum("bhts,bshd->bthd", pattern, v).reshape(batch, length, width)
        attention = activation("attn_out", self.attn_projection(attention))
        middle = activation("resid_mid", residual + attention)
        mlp = activation("mlp_out", self.mlp(self.ln_mlp(middle)))
        return activation("resid_post", middle + mlp)


class TinyTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.d_model % c.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.token_embedding = nn.Embedding(c.n_keys + c.n_values, c.d_model)
        self.position_embedding = nn.Embedding(c.seq_len, c.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model)
        self.unembedding = nn.Linear(c.d_model, c.n_values, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.config.embedding_std if isinstance(module, nn.Embedding) else 0.02
            nn.init.normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, patch=None, return_cache=False):
        patch = patch or {}
        valid_components = {"resid_pre", "q", "k", "v", "attn_out", "resid_mid",
                            "mlp_out", "resid_post", "attn_pattern"}
        for layer, component in patch:
            if not 0 <= layer < self.config.n_layers or component not in valid_components:
                raise KeyError(f"Unknown intervention site {(layer, component)}")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        residual = self.token_embedding(tokens) + self.position_embedding(positions)
        cache = {} if return_cache else None
        for layer, block in enumerate(self.blocks):
            residual = block(residual, layer, patch, cache)
        logits = self.unembedding(self.final_norm(residual))
        return (logits, cache) if return_cache else logits


def load_model(path):
    """Load an experiment checkpoint on CPU; returns (eval_model, checkpoint)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
