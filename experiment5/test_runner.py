"""Independent random-weight E5 tests: no trained inputs or checkpoints."""

import copy
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import ModelConfig, TinyTransformer, generate_batch
from experiment5.conditions import GROUPED, generate_conditions
from experiment5.runner import allowed_mask, masked_pattern, run_layout, validate_layout


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(959001)
        cls.model = TinyTransformer(ModelConfig(embedding_std=1.0)).eval()
        tokens, targets, metadata = generate_batch(7, torch.Generator().manual_seed(959101))
        cls.arrays = {"tokens": tokens, "targets": targets, "query_pair": metadata["query_pair"], "values": metadata["values"]}
        cls.small = {k: v[:2] for k, v in cls.arrays.items()}
        cls.grid = generate_conditions()
        cls.grouped = next(l for l in cls.grid["primary"] if l["layout_indices"] == GROUPED)

    def test_exact_independent_combinatorics_and_checked_in_spec(self):
        independently_valid = {p for p in itertools.permutations(range(8)) if all(p.index(k) < p.index(k+1) for k in (0, 2, 4, 6))}
        supplied = self.grid["primary"] + self.grid["boundary"]
        self.assertEqual({tuple(l["layout_indices"][:8]) for l in supplied}, independently_valid)
        self.assertEqual(len(independently_valid), 2520)
        self.assertEqual(len(self.grid["primary"]), 105)
        self.assertEqual(sum(len(l["conditions"]) for l in self.grid["primary"]), 1044)
        self.assertEqual(sum(c["id"].startswith("sham_") for l in self.grid["primary"] for c in l["conditions"]), 624)
        self.assertEqual(sum(not any(c["id"].startswith("sham_") for c in l["conditions"]) for l in self.grid["primary"]), 3)
        self.assertEqual(json.loads(Path("experiment5/conditions.json").read_text()), self.grid)

    def test_all_masks_causal_true_key_self_and_matched_degrees(self):
        for layout in self.grid["primary"] + self.grid["boundary"] + [self.grid["edge_panel"], self.grid["references"]]:
            validate_layout(layout, layout["conditions"])
            ids = layout["layout_indices"]
            for c in layout["conditions"]:
                support = allowed_mask(layout, c)
                self.assertFalse(support.triu(1).any())
                for i in range(4):
                    row = ids.index(2*i+1)
                    self.assertTrue(support[row, ids.index(2*i)])
                    self.assertTrue(support[row, row])
                    if layout["family"] == "primary" and c["kind"] == "keyguard":
                        self.assertEqual(sum(bool(support[row, ids.index(2*j)]) for j in range(4)), i+1)
                for row in [ids.index(2*i) for i in range(4)] + [8]:
                    self.assertTrue(torch.equal(support[row], torch.arange(9) <= row))

    def test_all_105_restoration_oracle_noop_and_untouched_keys(self):
        for layout in self.grid["primary"]:
            result = run_layout(self.model, self.small, layout)
            correct, oracle = result["correct"], result["oracle"]
            for metric in ("l1_value_max_error_native", "l1_query_max_error_native"):
                self.assertLess(float(correct[metric].max()), 1e-6, layout["id"])
            self.assertLess(float(oracle["class_probability_max_error_native"].max()), 1e-6)
            for part in ("key", "value", "query"):
                self.assertLess(float(oracle[f"l2_input_{part}_max_error_native"].max()), 1e-6)
            for condition_result in result.values():
                self.assertEqual(float(condition_result["l1_key_max_error_baseline"].max()), 0)
            for metric in result["unguarded"]:
                np.testing.assert_array_equal(result["unguarded"][metric], result["noop"][metric])

    def test_all_boundary_own_key_self_state_invariance(self):
        for layout in self.grid["boundary"]:
            selected = [c for c in layout["conditions"] if c["id"] == "own_key_self"]
            result = run_layout(self.model, self.small, layout, selected)["own_key_self"]
            self.assertLess(float(result["l1_value_max_error_own_key_self"].max()), 1e-6, layout["id"])
            self.assertLess(float(result["l1_query_max_error_native"].max()), 1e-6)
            self.assertEqual(float(result["l1_key_max_error_baseline"].max()), 0)

    def test_boundary_does_not_claim_native_restoration(self):
        # Own pairs are available, but V1 occurs before V0 and lacks its state.
        layout = next(l for l in self.grid["boundary"] if l["layout_indices"] == [0, 2, 3, 1, 4, 5, 6, 7, 8])
        result = run_layout(self.model, self.arrays, layout)
        self.assertGreater(float(result["logical_prefix"]["l1_value_max_error_native"].max()), 1e-5)
        own = result["own_key_self"]
        self.assertGreater(float(own["l1_value_max_error_native"].max()), 1e-5)
        self.assertLess(float(own["l1_value_max_error_own_key_self"].max()), 1e-6)

    def test_independent_live_score_mask_oracle(self):
        layouts = [self.grouped, self.grid["boundary"][17], self.grid["edge_panel"]]
        softmax = torch.Tensor.softmax
        for layout in layouts:
            for c in layout["conditions"]:
                if c.get("oracle") or c["kind"] in ("unguarded", "noop"):
                    continue
                # Construct forbidden physical entries independently of runner.
                ids = layout["layout_indices"]
                forbidden = []
                for dst, identity in enumerate(ids):
                    if identity >= 8 or identity % 2 == 0:
                        continue
                    i = identity//2
                    for src, token in enumerate(ids[:dst+1]):
                        drop = ((c["kind"] == "keyguard" and token % 2 == 0 and token < 8 and token//2 not in c["keep_keys"][i])
                                or (c["kind"] == "logical_prefix" and token > identity)
                                or (c["kind"] == "own_key_self" and token not in (identity-1, identity)))
                        if drop:
                            forbidden.append((dst, src))
                hits = []

                def direct(scores, dim=None, *args, **kwargs):
                    if scores.ndim == 4 and scores.shape[-2:] == (9, 9) and not hits:
                        hits.append(True)
                        scores = scores.clone()
                        for dst, src in forbidden:
                            scores[:, :, dst, src] = -torch.inf
                    return softmax(scores, dim, *args, **kwargs)

                tokens = self.arrays["tokens"][:, ids]
                with torch.inference_mode():
                    embedding = self.model.token_embedding(tokens) + self.model.position_embedding(torch.tensor(ids))
                    with patch.object(torch.Tensor, "softmax", direct):
                        logits = self.model(tokens, patch={(0, "resid_pre"): (slice(None), embedding)})
                expected = logits[:, -1].softmax(-1).numpy()
                observed = run_layout(self.model, self.arrays, layout, [c])[c["id"]]["class_probability"]
                self.assertEqual(len(hits), 1)
                np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)

    def test_masked_softmax_stable_when_retained_mass_underflows(self):
        q, k = torch.zeros(1, 9, 4, 16), torch.zeros(1, 9, 4, 16)
        q[:, 4, :, 0] = 100
        k[:, 1, :, 0] = 100
        scores = torch.einsum("bthd,bshd->bhts", q, k)/4
        scores.masked_fill_(torch.ones(9, 9, dtype=torch.bool).triu(1), -torch.inf)
        cache = {(0, "q"): q, (0, "k"): k, (0, "attn_pattern"): scores.softmax(-1)}
        correct = next(c for c in self.grouped["conditions"] if c["id"] == "correct")
        self.assertEqual(float(cache[(0, "attn_pattern")][:, :, 4, [0, 4]].sum()), 0)
        result = masked_pattern(cache, self.grouped, correct)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[:, :, 4, 0], torch.full((1, 4), .5))
        torch.testing.assert_close(result.sum(-1), torch.ones_like(result.sum(-1)))

    def test_edges_readd_exactly_one_connection_per_head(self):
        correct = next(c for c in self.grouped["conditions"] if c["id"] == "correct")
        base = allowed_mask(self.grouped, correct)
        self.assertEqual(len(self.grid["edge_panel"]["conditions"]), 6)
        for c in self.grid["edge_panel"]["conditions"]:
            mask = allowed_mask(self.grid["edge_panel"], c)
            change = torch.where(mask != base)
            i, j = c["edge"]
            self.assertEqual([(int(a), int(b)) for a, b in zip(*change)], [(4+i, j)])
            self.assertTrue(mask[4+i, j])

    def test_batching_shapes_and_invalid_metadata(self):
        a = run_layout(self.model, self.arrays, self.grouped, batch_size=7)
        b = run_layout(self.model, self.arrays, self.grouped, batch_size=3)
        for condition, metrics in a.items():
            for metric, value in metrics.items():
                self.assertEqual(value.shape, (7, 16) if metric == "class_probability" else (7,))
                np.testing.assert_allclose(value, b[condition][metric], rtol=1e-5, atol=1e-6)
        invalid = dict(self.arrays)
        invalid["targets"] = (invalid["targets"] + 1) % 16
        with self.assertRaises(ValueError):
            run_layout(self.model, invalid, self.grouped)
        malformed = copy.deepcopy(self.grouped)
        malformed["conditions"][2]["keep_keys"][0] = [1]
        with self.assertRaises(ValueError):
            run_layout(self.model, self.arrays, malformed)


if __name__ == "__main__":
    unittest.main()
