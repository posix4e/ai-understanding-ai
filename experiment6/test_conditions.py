"""Pure synthetic token-index tests; no pretrained model or tokenizer needed."""
import json
import unittest
from itertools import combinations, product

import torch

from experiment6.conditions import CONDITIONS, build_conditions, construct_mask
from experiment6.data import EncodedCase


def fake_case(prefix_length=3):
    chunks = [list(range(prefix_length + 2*i, prefix_length + 2*i + 2)) for i in range(8)]
    suffix = list(range(prefix_length+16, prefix_length+20))
    return EncodedCase(torch.arange(prefix_length+20), chunks, suffix,
                       list(range(prefix_length)), "synthetic index fixture")


class ConditionsTests(unittest.TestCase):
    def test_schema_and_exhaustive_grid(self):
        self.assertEqual(json.loads(json.dumps(CONDITIONS)), CONDITIONS)
        self.assertEqual(len(CONDITIONS), 15)
        self.assertEqual(len({c["id"] for c in CONDITIONS}), 15)
        self.assertEqual([c["id"] for c in CONDITIONS],
                         ["native", "grouped_physical", "grouped_canonical", "grouped_noop",
                          "guard_block0"] + [f"sham_{i}" for i in range(8)] + ["guard_block5", "guard_all"])
        guards = [c for c in CONDITIONS if c["family"] in ("primary_guard", "matched_sham")]
        expected = set(product(*[[choice for choice in combinations(range(4), i+1) if i in choice]
                                 for i in range(4)]))
        observed = {tuple(tuple(row) for row in c["retained_keys"]) for c in guards}
        self.assertEqual(observed, expected)
        self.assertEqual(len(expected), 9)
        self.assertEqual([(c["a"], c["b"]) for c in guards[1:]],
                         [(0, 0), (0, 1), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 3)])
        copy = build_conditions()
        copy[4]["retained_keys"][0].append(3)
        self.assertEqual(CONDITIONS[4]["retained_keys"][0], [0])

    def test_masks_edit_whole_key_chunks_only_and_match_row_degrees(self):
        for prefix_length in (0, 3, 15):
            encoded = fake_case(prefix_length)
            for condition in CONDITIONS:
                mask = construct_mask(encoded, condition)
                permutation = encoded.permutation(grouped=condition["grouped"])
                inverse = {int(original): physical for physical, original in enumerate(permutation)}
                self.assertEqual(mask.dtype, torch.bool)
                self.assertTrue(mask.diagonal().all())
                self.assertFalse(mask.triu(1).any())
                keep = condition["retained_keys"]
                for row in range(mask.shape[0]):
                    original = int(permutation[row])
                    value_pair = next((i for i in range(4) if original in encoded.chunks[2*i+1]), None)
                    for column in range(mask.shape[1]):
                        source = int(permutation[column])
                        source_pair = next((j for j in range(4) if source in encoded.chunks[2*j]), None)
                        deleted = (keep is not None and value_pair is not None
                                   and source_pair is not None and source_pair not in keep[value_pair])
                        self.assertEqual(bool(mask[row, column]), column <= row and not deleted)
                if keep is not None:
                    for i in range(4):
                        key_columns = [inverse[p] for j in range(4) for p in encoded.chunks[2*j]]
                        for original in encoded.chunks[2*i+1]:
                            self.assertEqual(int(mask[inverse[original], key_columns].sum()), 2*(i+1))

    def test_correct_guard_retains_exact_native_predecessor_sets_at_values(self):
        encoded = fake_case()
        perm = encoded.permutation(grouped=True)
        mask = construct_mask(encoded, "guard_block0")
        for row, original in enumerate(perm.tolist()):
            if any(original in encoded.chunks[2*i+1] for i in range(4)):
                retained_original_ids = set(perm[mask[row]].tolist())
                self.assertEqual(retained_original_ids, set(range(original+1)))

    def test_noop_and_secondary_layer_scope(self):
        encoded = fake_case()
        full = torch.ones(encoded.ids.numel(), encoded.ids.numel(), dtype=torch.bool).tril()
        for identifier in ("native", "grouped_physical", "grouped_canonical", "grouped_noop"):
            self.assertTrue(torch.equal(construct_mask(encoded, identifier), full))
        spec = {c["id"]: c for c in CONDITIONS}
        self.assertEqual(spec["grouped_noop"]["layers"], [0])
        self.assertEqual(spec["guard_block5"]["layers"], [5])
        self.assertEqual(spec["guard_all"]["layers"], list(range(12)))
        for identifier in ("guard_block5", "guard_all"):
            self.assertTrue(torch.equal(construct_mask(encoded, identifier), construct_mask(encoded, "guard_block0")))

    def test_invalid_own_key_or_key_chunk_rejected(self):
        encoded = fake_case()
        invalid = build_conditions()[4]
        invalid["retained_keys"][1] = [0, 2]
        with self.assertRaisesRegex(ValueError, "own key"):
            construct_mask(encoded, invalid)
        encoded.chunks[0] = encoded.chunks[0][:1]
        with self.assertRaisesRegex(ValueError, "two tokens"):
            construct_mask(encoded, "guard_block0")
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            construct_mask(fake_case(), "unknown")


if __name__ == "__main__":
    unittest.main()
