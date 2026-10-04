"""Random-weight tests only; no pretrained weights or E6 outputs are loaded."""
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiment7.model_adapter import GPT2DiagnosticAdapter, ResidualPatch


class DiagnosticAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17001)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=73, n_positions=48, n_embd=32,
                                         n_layer=3, n_head=4, attn_implementation="eager"))
        original = {name: parameter.clone() for name, parameter in model.named_parameters()}
        self.adapter = GPT2DiagnosticAdapter(model)
        self.model = self.adapter.model
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter.float(), original[name]))
        self.tokens = torch.randint(0, 73, (3, 20))
        self.physical = torch.ones(20, 20, dtype=torch.bool).tril()
        # Prefix(2), K0(2), V0(2), ..., K3(2), V3(2), query suffix(2).
        self.keys = [[2+4*i, 3+4*i] for i in range(4)]
        self.values = [[4+4*i, 5+4*i] for i in range(4)]
        self.permutation = torch.tensor([0,1] + sum(self.keys,[]) + sum(self.values,[]) + [18,19])
        self.transport = self.physical[self.permutation][:,self.permutation]

    def grouped(self, **kwargs):
        return self.adapter(self.tokens[:,self.permutation], self.permutation.expand(3,-1), **kwargs)

    def test_native_matches_stock_and_exact_promotion(self):
        with torch.inference_mode():
            stock = self.model(self.tokens, use_cache=False).logits[:,-1]
        result = self.adapter(self.tokens, capture_layers="all")
        torch.testing.assert_close(result.logits, stock, atol=2e-14, rtol=2e-13)
        self.assertEqual(result.logits.dtype, torch.float64)
        self.assertEqual(set(result.post_block), {0,1,2})
        self.assertTrue(all(v.dtype == torch.float64 for v in result.post_block.values()))
        self.assertFalse(result.used_future_edges)

    def test_full_transport_recovers_every_native_block(self):
        native = self.adapter(self.tokens, capture_layers="all")
        baseline = self.grouped(capture_layers="all")
        oracle = self.grouped(layer_masks={i:self.transport for i in range(3)},
                              allow_future=True, capture_layers="all")
        self.assertTrue(oracle.used_future_edges)
        self.assertTrue(self.transport.triu(1).any())
        for layer in range(3):
            torch.testing.assert_close(oracle.post_block[layer], native.post_block[layer][:,self.permutation],
                                       atol=2e-14, rtol=2e-13)
        torch.testing.assert_close(oracle.logits, native.logits, atol=2e-14, rtol=2e-13)
        self.assertGreater((baseline.logits-native.logits).abs().max().item(), 1e-7)

    def test_transport_decomposes_into_edge_addition_and_deletion(self):
        deletion = self.physical & self.transport
        addition = self.physical | self.transport
        added_edges = addition & ~self.physical
        removed_edges = self.physical & ~self.transport
        self.assertTrue(added_edges.any())
        self.assertTrue(removed_edges.any())
        self.assertFalse(deletion.triu(1).any())
        self.assertTrue(torch.equal(deletion | added_edges, self.transport))
        self.assertTrue(torch.equal(addition & ~removed_edges, self.transport))
        via_parts = self.grouped(layer_masks={i:deletion | added_edges for i in range(3)}, allow_future=True)
        native = self.adapter(self.tokens)
        torch.testing.assert_close(via_parts.logits, native.logits, atol=2e-14, rtol=2e-13)

    def test_future_edges_require_opt_in_and_really_open(self):
        with self.assertRaisesRegex(ValueError, "explicit oracle"):
            self.grouped(layer_masks={0:self.transport})
        seen = {}
        handles = []
        for layer, block in enumerate(self.model.transformer.h):
            def capture(module, args, output, layer=layer):
                seen[layer] = output[1].clone()
            handles.append(block.attn.register_forward_hook(capture))
        try:
            self.grouped(layer_masks={0:self.transport}, allow_future=True)
        finally:
            for handle in handles:
                handle.remove()
        # K1 now physically precedes V0, which was in its original past.
        self.assertTrue((seen[0][:,:,4,10] > 0).all())
        self.assertTrue((seen[1][:,:,4,10] == 0).all())
        self.assertTrue((seen[2][:,:,4,10] == 0).all())
        self.assertTrue((seen[0].masked_select(~self.transport[None,None]) == 0).all())
        for attention in seen.values():
            self.assertEqual(attention.dtype, torch.float64)
            torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-14, rtol=0)

    def test_primary_block0_guard_restores_values_and_suffix_only(self):
        # The physical/native intersection deletes only future-logical keys
        # from value rows in this grouped serialization.
        guard = self.physical.clone()
        for i in range(4):
            for row in (10+2*i,11+2*i):
                for j in range(i+1,4):
                    guard[row,2+2*j:4+2*j] = False
        native = self.adapter(self.tokens, capture_layers=[0])
        baseline = self.grouped(capture_layers=[0])
        guarded = self.grouped(layer_masks={0:guard}, capture_layers=[0])
        torch.testing.assert_close(guarded.first_block[:,10:], native.first_block[:,self.permutation][:,10:],
                                   atol=2e-14, rtol=2e-13)
        torch.testing.assert_close(guarded.first_block[:,:10], baseline.first_block[:,:10], atol=0, rtol=0)
        self.assertGreater((guarded.logits-native.logits).abs().max().item(),1e-7)

    def test_all_residual_patches_and_all_mask_noops(self):
        baseline = self.adapter(self.tokens, capture_layers="all")
        patches = {layer:ResidualPatch(torch.arange(20),state) for layer,state in baseline.post_block.items()}
        patched = self.adapter(self.tokens, layer_masks={i:self.physical for i in range(3)},
                               residual_patches=patches, capture_layers="all")
        torch.testing.assert_close(patched.logits,baseline.logits,atol=0,rtol=0)
        for layer in range(3):
            torch.testing.assert_close(patched.post_block[layer],baseline.post_block[layer],atol=0,rtol=0)

    def test_selected_residual_rows_and_batch_specific_masks(self):
        baseline = self.adapter(self.tokens,capture_layers=[0])
        rows = torch.tensor([2,7])
        donor = baseline.first_block[:,rows] + torch.linspace(-.3,.4,32,dtype=torch.float64)
        changed = self.adapter(self.tokens,residual_patches={0:ResidualPatch(rows,donor)},capture_layers=[0])
        torch.testing.assert_close(changed.first_block[:,rows],donor,atol=0,rtol=0)
        untouched = [i for i in range(20) if i not in rows.tolist()]
        torch.testing.assert_close(changed.first_block[:,untouched],baseline.first_block[:,untouched],atol=0,rtol=0)
        self.assertGreater((changed.logits-baseline.logits).abs().max().item(),1e-7)
        masks = self.physical.repeat(3,1,1)
        masks[0,19,:10] = False
        batched = self.adapter(self.tokens,layer_masks={1:masks})
        for row in range(3):
            single = self.adapter(self.tokens[row:row+1],layer_masks={1:masks[row]})
            torch.testing.assert_close(batched.logits[row:row+1],single.logits,atol=2e-14,rtol=2e-13)

    def test_parameters_buffers_and_hooks_restored_on_success_and_exception(self):
        params = {name:p.clone() for name,p in self.model.named_parameters()}
        buffers = {name:(b,b.clone()) for name,b in self.model.named_buffers()}
        hook_counts = [(len(block.attn._forward_pre_hooks),len(block._forward_hooks)) for block in self.model.transformer.h]
        baseline = self.adapter(self.tokens)
        self.grouped(layer_masks={i:self.transport for i in range(3)},allow_future=True,capture_layers="all")
        def fail(module,args,output):
            raise RuntimeError("synthetic mid-forward failure")
        failure_handle = self.model.transformer.h[1].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,"synthetic mid-forward failure"):
                self.grouped(layer_masks={i:self.transport for i in range(3)},allow_future=True,capture_layers="all")
        finally:
            failure_handle.remove()
        self.assertFalse(self.adapter._active)
        for name,p in self.model.named_parameters():
            self.assertTrue(torch.equal(p,params[name]))
        for name,b in self.model.named_buffers():
            self.assertIs(b,buffers[name][0])
            self.assertTrue(torch.equal(b,buffers[name][1]))
        self.assertEqual(hook_counts,[(len(b.attn._forward_pre_hooks),len(b._forward_hooks)) for b in self.model.transformer.h])
        torch.testing.assert_close(self.adapter(self.tokens).logits,baseline.logits,atol=0,rtol=0)

    def test_invalid_masks_and_patches_rejected(self):
        with self.assertRaisesRegex(ValueError,"at least one source"):
            self.adapter(self.tokens,layer_masks={0:torch.zeros(20,20,dtype=torch.bool)})
        with self.assertRaisesRegex(TypeError,"Boolean"):
            self.adapter(self.tokens,layer_masks={0:torch.ones(20,20)})
        with self.assertRaisesRegex(ValueError,"unique"):
            self.adapter(self.tokens,residual_patches={0:ResidualPatch(torch.tensor([1,1]),torch.zeros(3,2,32,dtype=torch.float64))})
        with self.assertRaisesRegex(ValueError,"float64"):
            self.adapter(self.tokens,residual_patches={0:ResidualPatch(torch.tensor([1]),torch.zeros(3,1,32))})
        with self.assertRaisesRegex(ValueError,"block index"):
            self.adapter(self.tokens,layer_masks={3:self.physical})
        self.adapter._active = True
        try:
            with self.assertRaisesRegex(RuntimeError,"overlap"):
                self.adapter(self.tokens)
        finally:
            self.adapter._active = False


if __name__ == "__main__":
    unittest.main()
