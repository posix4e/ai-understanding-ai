"""Independent E3 synthetic checks; no trained checkpoints or E3 inputs read."""

import copy
import json
from pathlib import Path
import unittest

import torch

from model import ModelConfig, TinyTransformer, generate_batch
from experiment3.conditions import generate_conditions
from experiment3.runner import run, validate_conditions


class CoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(93017)
        cls.model = TinyTransformer(ModelConfig(embedding_std=1.0)).eval()
        tokens, targets, metadata = generate_batch(13, torch.Generator().manual_seed(93018))
        cls.arrays = {"tokens": tokens, "targets": targets,
                      "query_pair": metadata["query_pair"], "values": metadata["values"]}
        cls.conditions = generate_conditions()
        cls.by_id = {condition["id"]: condition for condition in cls.conditions}
        cls.results = run(cls.model, cls.arrays, cls.conditions, batch_size=13)

    def test_complete_permutation_enumeration(self):
        self.assertEqual(len(self.conditions), 50)
        self.assertEqual(len(self.by_id), 50)
        self.assertEqual(sum(c["family"] == "coherent" for c in self.conditions), 23)
        self.assertEqual(sum(c["family"] == "value_only" for c in self.conditions), 23)
        self.assertEqual(sum(c["primary_group"] == "coherent_control" for c in self.conditions), 9)
        self.assertEqual(sum(c["primary_group"] == "value_derangement" for c in self.conditions), 9)
        self.assertEqual(json.loads(Path("experiment3/conditions.json").read_text()), self.conditions)

    def test_slot_map_follows_assigned_coordinates(self):
        for condition in self.conditions:
            if not condition["slot_mapping_defined"]:
                continue
            layout = condition["layout_indices"]
            position_ids = condition["position_ids"] or list(range(9))
            key_ids = [position_ids[layout.index(2 * i)] for i in range(4)]
            value_ids = [position_ids[layout.index(2 * i + 1)] for i in range(4)]
            for value in range(4):
                expected_key = key_ids.index(value_ids[value] - 1)
                self.assertEqual(condition["slot_key_by_value"][value], expected_key)
            for query in range(4):
                expected_value = value_ids.index(key_ids[query] + 1)
                self.assertEqual(condition["slot_value_by_query"][query], expected_value)
            if condition["family"] == "coherent":
                self.assertEqual(condition["slot_value_by_query"], [0, 1, 2, 3])
            if condition["primary_group"] == "value_derangement":
                self.assertTrue(all(q != v for q, v in enumerate(condition["slot_value_by_query"])))

    def test_identity_position_patch_is_exact_noop(self):
        self.assertEqual(self.results["original_native"], self.results["original_noop"])

    def test_grouped_native_slots_explicitly_undefined(self):
        self.assertFalse(self.by_id["grouped_native"]["slot_mapping_defined"])
        results = self.results["grouped_native"]
        self.assertEqual(results["original_probability"], results["slot_probability"])
        self.assertEqual(results["original_accuracy"], results["slot_accuracy"])

    def test_runner_against_direct_coordinate_and_attention_oracle(self):
        # Manually specify pi=(2,0,1,3), avoiding the generated coordinate arrays.
        ids = torch.tensor([0, 2, 4, 6, 5, 1, 3, 7, 8])
        tokens = self.arrays["tokens"][:, [0, 2, 4, 6, 1, 3, 5, 7, 8]]
        with torch.no_grad():
            residual = self.model.token_embedding(tokens) + self.model.position_embedding(ids)
            logits, cache = self.model(tokens, patch={(0, "resid_pre"): (slice(None), residual)}, return_cache=True)
        self.assertTrue(torch.equal(cache[(0, "resid_pre")], residual))
        probability = logits[:, -1].softmax(-1)
        actual = self.results["value_2013"]
        inverse_pi = [1, 2, 0, 3]
        key_for_value = [2, 0, 1, 3]
        for row in range(len(tokens)):
            q = int(self.arrays["query_pair"][row])
            target = int(self.arrays["targets"][row])
            slot_pair = inverse_pi[q]
            slot_target = int(self.arrays["values"][row, slot_pair])
            self.assertEqual(actual["original_probability"][row], float(probability[row, target]))
            self.assertEqual(actual["slot_probability"][row], float(probability[row, slot_target]))
            self.assertEqual(actual["argmax_class"][row], int(logits[row, -1].argmax()))
            for head in range(4):
                true_attention = sum(float(cache[(0, "attn_pattern")][row, head, 4 + value, value]) for value in range(4)) / 4
                slot_attention = sum(float(cache[(0, "attn_pattern")][row, head, 4 + value, key_for_value[value]]) for value in range(4)) / 4
                self.assertAlmostEqual(actual["l1_true_key_attention"][row][head], true_attention, places=7)
                self.assertAlmostEqual(actual["l1_slot_key_attention"][row][head], slot_attention, places=7)
                self.assertEqual(actual["l2_original_value_attention"][row][head], float(cache[(1, "attn_pattern")][row, head, 8, 4 + q]))
                self.assertEqual(actual["l2_slot_value_attention"][row][head], float(cache[(1, "attn_pattern")][row, head, 8, 4 + slot_pair]))
        for layer in (0, 1):
            self.assertEqual(float(cache[(layer, "attn_pattern")].triu(1).abs().max()), 0.0)

    def test_heads_kept_separate_and_all_values_averaged(self):
        for results in self.results.values():
            probability = torch.tensor(results["class_probability"])
            self.assertEqual(tuple(probability.shape), (13, 16))
            torch.testing.assert_close(probability.sum(1), torch.ones(13), rtol=1e-6, atol=1e-7)
            self.assertEqual(probability.argmax(1).tolist(), results["argmax_class"])
            for metric in ("l1_true_key_attention", "l1_slot_key_attention", "l2_original_value_attention", "l2_slot_value_attention"):
                self.assertEqual(tuple(torch.tensor(results[metric]).shape), (13, 4))
                values = torch.tensor(results[metric])
                self.assertTrue(torch.all((values >= 0) & (values <= 1)))

    def test_batching_invariance(self):
        selected = [self.by_id[name] for name in ("original_native", "grouped_canonical", "coherent_1032", "value_1032")]
        results = run(self.model, self.arrays, selected, batch_size=4)
        for name, metrics in results.items():
            for metric, values in metrics.items():
                torch.testing.assert_close(torch.tensor(values), torch.tensor(self.results[name][metric]), rtol=1e-6, atol=1e-7)

    def test_bad_coordinates_and_metadata_rejected(self):
        condition = copy.deepcopy(self.by_id["grouped_canonical"])
        condition["position_ids"][0] = 8
        with self.assertRaises(ValueError):
            validate_conditions([condition])
        bad_arrays = dict(self.arrays)
        bad_arrays["targets"] = (self.arrays["targets"] + 1) % 16
        with self.assertRaises(ValueError):
            run(self.model, bad_arrays, [self.by_id["original_native"]])


if __name__ == "__main__":
    unittest.main()
