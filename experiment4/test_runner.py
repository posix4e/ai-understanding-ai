"""E4 independent random-weight fixtures; never read trained data/checkpoints."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from model import ModelConfig, TinyTransformer, generate_batch
from experiment4.conditions import GROUPED, generate_conditions
from experiment4.runner import guarded_pattern, run, validate_conditions


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(95001)
        cls.model = TinyTransformer(ModelConfig(embedding_std=1.0)).eval()
        tokens, targets, meta = generate_batch(13, torch.Generator().manual_seed(95101))
        cls.arrays = {"tokens": tokens, "targets": targets, "query_pair": meta["query_pair"], "values": meta["values"]}
        cls.conditions = generate_conditions()
        cls.by_id = {condition["id"]: condition for condition in cls.conditions}
        cls.results = run(cls.model, cls.arrays, cls.conditions, batch_size=13)

    def test_all_fifteen_conditions_and_nine_matched_masks(self):
        self.assertEqual(len(self.conditions), 15)
        self.assertEqual(len(self.by_id), 15)
        guards = [c for c in self.conditions if c["family"] == "matched_guard"]
        self.assertEqual(len(guards), 9)
        self.assertEqual(len({tuple(tuple(keys) for keys in c["guard_keep_keys"]) for c in guards}), 9)
        for condition in guards:
            mask = condition["guard_keep_keys"]
            self.assertEqual([len(keys) for keys in mask], [1, 2, 3, 4])
            self.assertTrue(all(i in keys for i, keys in enumerate(mask)))
            self.assertEqual(sum(4 - len(keys) for keys in mask), 6)
        self.assertEqual(self.by_id["guard_a0_b3"]["guard_keep_keys"], [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
        self.assertEqual(json.loads(Path("experiment4/conditions.json").read_text()), self.conditions)

    def test_grouped_attention_noop_is_exact(self):
        self.assertEqual(self.results["grouped_canonical"], self.results["grouped_noop"])

    def test_theorem_restores_values_and_query_but_not_keys(self):
        correct = self.results["guard_a0_b3"]
        self.assertLess(max(correct["l1_value_max_error_native"]), 1e-6)
        self.assertLess(max(correct["l1_query_max_error_native"]), 1e-6)
        self.assertEqual(max(correct["l1_key_max_error_grouped"]), 0)
        self.assertGreater(max(correct["l1_key_max_error_native"]), 1e-5)
        for condition in self.conditions:
            if condition["family"] == "matched_guard":
                self.assertEqual(max(self.results[condition["id"]]["l1_key_max_error_grouped"]), 0)

    def test_guard_matches_value_oracle_and_full_oracle_matches_native(self):
        for first, second in [("guard_a0_b3", "restore_values"), ("guard_restore_keys", "native")]:
            torch.testing.assert_close(torch.tensor(self.results[first]["class_probability"]),
                                       torch.tensor(self.results[second]["class_probability"]), rtol=1e-6, atol=1e-7)
        full = self.results["guard_restore_keys"]
        for component in ("value", "query", "key"):
            self.assertLess(max(full[f"l2_input_{component}_max_error_native"]), 1e-6)

    def test_actual_l1_and_effective_l2_transplants_are_distinct(self):
        restored = self.results["restore_values"]
        self.assertGreater(max(restored["l1_value_max_error_native"]), 1e-5)
        self.assertEqual(max(restored["l2_input_value_max_error_native"]), 0)
        self.assertEqual(max(restored["l2_input_key_max_error_grouped"]), 0)
        keys = self.results["restore_keys"]
        self.assertGreater(max(keys["l1_key_max_error_native"]), 1e-5)
        self.assertEqual(max(keys["l2_input_key_max_error_native"]), 0)

    def test_cached_guard_matches_independent_live_logit_mask(self):
        tokens = self.arrays["tokens"][:, GROUPED]
        with torch.no_grad():
            embedding = self.model.token_embedding(tokens) + self.model.position_embedding(torch.tensor(GROUPED))
        original_softmax = torch.Tensor.softmax
        for name in ("guard_a0_b3", "guard_a3_b0"):
            condition = self.by_id[name]
            intercepted = []

            def live_mask(scores, dim=None, *args, **kwargs):
                if scores.ndim == 4 and scores.shape[-2:] == (9, 9) and not intercepted:
                    intercepted.append(True)
                    scores = scores.clone()
                    # Independent direct mask on the actual model scores,
                    # avoiding cached Q/K and guarded_pattern entirely.
                    a, b = condition["a"], condition["b"]
                    retain = [{0}, {1, a}, set(range(4)) - {b}, set(range(4))]
                    for i in range(4):
                        for j in range(4):
                            if j not in retain[i]:
                                scores[:, :, 4 + i, j] = -torch.inf
                return original_softmax(scores, dim, *args, **kwargs)

            with torch.no_grad(), patch.object(torch.Tensor, "softmax", live_mask):
                logits = self.model(tokens, patch={(0, "resid_pre"): (slice(None), embedding)})
            self.assertEqual(len(intercepted), 1)
            expected = logits[:, -1].softmax(-1)
            torch.testing.assert_close(torch.tensor(self.results[name]["class_probability"]), expected, rtol=1e-6, atol=1e-7)

    def test_masks_preserve_all_unselected_rows_and_causality(self):
        tokens = self.arrays["tokens"][:, GROUPED]
        with torch.no_grad():
            embedding = self.model.token_embedding(tokens) + self.model.position_embedding(torch.tensor(GROUPED))
            _, cache = self.model(tokens, patch={(0, "resid_pre"): (slice(None), embedding)}, return_cache=True)
        for condition in self.conditions:
            if condition["family"] != "matched_guard":
                continue
            pattern = guarded_pattern(cache, condition["guard_keep_keys"])
            self.assertTrue(torch.equal(pattern[:, :, [0, 1, 2, 3, 7, 8]], cache[(0, "attn_pattern")][:, :, [0, 1, 2, 3, 7, 8]]))
            self.assertEqual(float(pattern.triu(1).abs().max()), 0)
            torch.testing.assert_close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), rtol=1e-6, atol=1e-7)
            for value, keep in enumerate(condition["guard_keep_keys"]):
                for key in range(4):
                    if key not in keep:
                        self.assertEqual(float(pattern[:, :, 4 + value, key].abs().max()), 0)

    def test_stable_mask_handles_underflowed_retained_probability(self):
        # Construct finite scores with all pre-mask mass on a removed key.
        q = torch.zeros(1, 9, 4, 16)
        k = torch.zeros_like(q)
        q[:, 4, :, 0] = 100
        k[:, 1, :, 0] = 100
        score = torch.einsum("bthd,bshd->bhts", q, k) / 4
        score.masked_fill_(torch.ones(9, 9, dtype=torch.bool).triu(1), -torch.inf)
        cache = {(0, "q"): q, (0, "k"): k, (0, "attn_pattern"): score.softmax(-1)}
        self.assertEqual(float(cache[(0, "attn_pattern")][:, :, 4, [0, 4]].sum()), 0)
        result = guarded_pattern(cache, self.by_id["guard_a0_b3"]["guard_keep_keys"])
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[:, :, 4, 0], torch.full((1, 4), .5)))
        self.assertTrue(torch.equal(result[:, :, 4, 4], torch.full((1, 4), .5)))

    def test_batching_and_all_head_diagnostics(self):
        chosen = [self.by_id[name] for name in ("native", "guard_a0_b3", "restore_values", "guard_restore_keys")]
        results = run(self.model, self.arrays, chosen, batch_size=4)
        for name, metrics in results.items():
            for metric, values in metrics.items():
                torch.testing.assert_close(torch.tensor(values), torch.tensor(self.results[name][metric]), rtol=1e-5, atol=1e-6)
            for metric in ("l1_true_key_attention", "l2_correct_value_attention"):
                self.assertEqual(tuple(torch.tensor(metrics[metric]).shape), (13, 4))

    def test_invalid_masks_and_metadata_rejected(self):
        condition = copy.deepcopy(self.by_id["guard_a0_b3"])
        condition["guard_keep_keys"][0] = [1]
        with self.assertRaises(ValueError):
            validate_conditions([condition])
        arrays = dict(self.arrays)
        arrays["targets"] = (arrays["targets"] + 1) % 16
        with self.assertRaises(ValueError):
            run(self.model, arrays, [self.by_id["native"]])


if __name__ == "__main__":
    unittest.main()
