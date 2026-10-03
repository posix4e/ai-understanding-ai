"""Offline random-weight tests only: python -m unittest experiment6.test_adapter."""
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiment6.model_adapter import GPT2Adapter, causal_allowed


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1601)
        config = GPT2Config(vocab_size=71, n_positions=48, n_embd=32,
                            n_layer=3, n_head=4, resid_pdrop=0.1,
                            embd_pdrop=0.1, attn_pdrop=0.1,
                            attn_implementation="eager")
        self.model = GPT2LMHeadModel(config).eval()
        self.adapter = GPT2Adapter(self.model)
        self.tokens = torch.randint(0, 71, (3, 20))

    def test_stock_noop_and_unchanged_weights(self):
        weights = {name: value.clone() for name, value in self.model.state_dict().items()}
        with torch.inference_mode():
            stock = self.model(self.tokens, use_cache=False).logits[:, -1]
        plain = self.adapter(self.tokens)
        noop = self.adapter(self.tokens, layer_masks={i: torch.ones(20, 20, dtype=torch.bool) for i in range(3)})
        torch.testing.assert_close(plain.logits, stock, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(noop.logits, plain.logits, atol=0, rtol=0)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]), name)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertFalse(self.model.training)

    def test_future_edges_cannot_be_opened_and_masks_are_layer_scoped(self):
        observed = {}
        handles = []
        for layer, block in enumerate(self.model.transformer.h):
            def capture(module, args, kwargs, output, layer=layer):
                observed[layer] = output[1].clone()
            handles.append(block.attn.register_forward_hook(capture, with_kwargs=True))
        try:
            allowed = torch.ones(20, 20, dtype=torch.bool)
            allowed[10, 2] = False
            self.adapter(self.tokens, layer_masks={1: allowed})
        finally:
            for handle in handles:
                handle.remove()
        for attention in observed.values():
            self.assertTrue(torch.equal(attention.triu(1), torch.zeros_like(attention)))
            torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
        self.assertTrue((observed[1][:, :, 10, 2] == 0).all())
        self.assertTrue((observed[0][:, :, 10, 2] > 0).all())
        self.assertTrue((observed[2][:, :, 10, 2] > 0).all())
        baseline = self.adapter(self.tokens, capture_first_block=True)
        changed = self.adapter(self.tokens, layer_masks={1: allowed}, capture_first_block=True)
        torch.testing.assert_close(baseline.first_block, changed.first_block, atol=0, rtol=0)
        self.assertGreater((baseline.logits - changed.logits).abs().max().item(), 1e-8)

    def test_two_token_chunks_first_block_restoration(self):
        # Prefix has 2 tokens. Each key and each value is a 2-token chunk.
        # Native: prefix K0 V0 K1 V1 K2 V2 K3 V3 query(2 tokens).
        keys = [[2 + 4*i, 3 + 4*i] for i in range(4)]
        values = [[4 + 4*i, 5 + 4*i] for i in range(4)]
        permutation = torch.tensor([0, 1] + sum(keys, []) + sum(values, []) + [18, 19])
        positions = permutation.expand(3, -1)
        grouped = self.tokens[:, permutation]
        allowed = torch.ones(20, 20, dtype=torch.bool).tril()
        for i in range(4):
            for query in (10 + 2*i, 11 + 2*i):
                for j in range(i+1, 4):
                    allowed[query, 2 + 2*j:4 + 2*j] = False
        native = self.adapter(self.tokens, capture_first_block=True)
        baseline = self.adapter(grouped, positions, capture_first_block=True)
        guarded = self.adapter(grouped, positions, {0: allowed}, capture_first_block=True)
        native_permuted = native.first_block[:, permutation]
        torch.testing.assert_close(guarded.first_block[:, 10:], native_permuted[:, 10:], atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(guarded.first_block[:, :10], baseline.first_block[:, :10], atol=0, rtol=0)
        self.assertGreater((baseline.first_block[:, 10:18] - native_permuted[:, 10:18]).abs().max().item(), 1e-5)
        # First-block identity does not imply deeper output identity.
        self.assertGreater((guarded.logits - native.logits).abs().max().item(), 1e-7)

    def test_batch_specific_masks(self):
        allowed = torch.ones(3, 20, 20, dtype=torch.bool)
        allowed[0, 19, 0:10] = False
        batched = self.adapter(self.tokens, layer_masks={0: allowed}).logits
        for row in range(3):
            separate = self.adapter(self.tokens[row:row+1], layer_masks={0: allowed[row]}).logits
            torch.testing.assert_close(batched[row:row+1], separate, atol=2e-7, rtol=2e-6)

    def test_invalid_masks_and_positions(self):
        with self.assertRaisesRegex(ValueError, "physical predecessor"):
            self.adapter(self.tokens, layer_masks={0: torch.zeros(20, 20, dtype=torch.bool)})
        future_only = torch.ones(20, 20, dtype=torch.bool).triu(1)
        with self.assertRaisesRegex(ValueError, "physical predecessor"):
            causal_allowed(future_only, 3, 20)
        for bad in (torch.ones(20, 20), torch.ones(19, 20, dtype=torch.bool)):
            with self.assertRaises((TypeError, ValueError)):
                self.adapter(self.tokens, layer_masks={0: bad})
        with self.assertRaisesRegex(ValueError, "layer"):
            self.adapter(self.tokens, layer_masks={3: torch.ones(20, 20, dtype=torch.bool)})
        with self.assertRaisesRegex(ValueError, "position"):
            self.adapter(self.tokens, position_ids=torch.full_like(self.tokens, 48))

    def test_hook_cleanup_after_failure(self):
        block = self.model.transformer.h[0]
        before = (len(block.attn._forward_pre_hooks), len(block._forward_hooks))
        with patch.object(self.model.transformer, "forward", side_effect=RuntimeError("synthetic")):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                self.adapter(self.tokens, layer_masks={0: torch.ones(20, 20, dtype=torch.bool)}, capture_first_block=True)
        self.assertEqual(before, (len(block.attn._forward_pre_hooks), len(block._forward_hooks)))
        self.assertFalse(self.adapter._active)
        self.assertTrue(torch.isfinite(self.adapter(self.tokens).logits).all())


if __name__ == "__main__":
    unittest.main()
