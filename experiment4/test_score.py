"""Synthetic E4 scoring fixtures only; no trained models or experimental data."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from experiment4.conditions import generate_conditions
from experiment4.score import ATTENTION_METRICS, STATE_METRICS, ci, lower_exceeds, score_seed, stratified_weights


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conditions = generate_conditions()
        cls.protocol = json.loads(Path("experiment4/protocol.json").read_text())
        cls.query = np.tile(np.arange(4), 4)
        cls.arrays = {"targets": cls.query.copy(), "query_pair": cls.query}
        cls.weights = stratified_weights(cls.query, 128, 8200001)
        cls.raw = {}
        cls.wrong = [c["id"] for c in cls.conditions if c["family"] == "matched_guard" and c["id"] != "guard_a0_b3"]
        for condition in cls.conditions:
            name = condition["id"]
            p = .99 if name in ("native", "guard_restore_keys") else .98 if name in ("guard_a0_b3", "restore_values") else .4 if name in ("grouped_canonical", "grouped_noop", "restore_keys") else .7
            cls.set_probability(cls.raw, name, np.full(16, p))
            for metric in STATE_METRICS:
                cls.raw[name + "/" + metric] = np.zeros(16)
            for metric in ATTENTION_METRICS:
                cls.raw[name + "/" + metric] = np.full((16, 4), 1 / 9)
        cls.forecasts = {metric: {c["id"]: float(cls.raw[c["id"] + "/" + metric].mean()) for c in cls.conditions}
                         for metric in ("original_probability", "accuracy")}

    @classmethod
    def set_probability(cls, raw, name, target_probability, competitor=15):
        full = np.zeros((16, 16))
        full[np.arange(16), cls.query] = target_probability
        full[:, competitor] = 1 - target_probability
        predicted = full.argmax(1)
        raw[name + "/class_probability"] = full
        raw[name + "/original_probability"] = np.asarray(target_probability)
        raw[name + "/argmax_class"] = predicted
        raw[name + "/accuracy"] = (predicted == cls.query).astype(int)

    def score(self, raw):
        return score_seed(raw, self.arrays, self.conditions, self.forecasts, self.protocol, self.weights)[0]

    def test_high_confidence_fixture_passes_every_gate(self):
        result = self.score(self.raw)
        self.assertTrue(all(result["validity_gates"].values()))
        self.assertTrue(all(result["recovery_gates"].values()))
        self.assertTrue(all(result["specificity_gates"].values()))
        self.assertEqual(result["forecast_statistics"]["all_15"]["count"], 30)
        self.assertEqual(result["forecast_statistics"]["matched_guards_9"]["count"], 18)
        self.assertEqual(result["forecast_statistics"]["oracles_3"]["count"], 6)
        self.assertEqual(result["forecast_statistics"]["references_3"]["count"], 6)

    def test_bootstrap_is_stratified_and_shared(self):
        np.testing.assert_allclose(self.weights.sum(1), 1)
        for q in range(4):
            np.testing.assert_allclose(self.weights[:, self.query == q].sum(1), .25)
        np.testing.assert_array_equal(self.weights, stratified_weights(self.query, 128, 8200001))
        x = np.arange(16) / 16
        np.testing.assert_allclose(self.weights @ x + self.weights @ (1 - x), 1)

    def test_full_class_oracle_check_detects_wrong_nontarget_mass(self):
        raw = copy.deepcopy(self.raw)
        self.set_probability(raw, "restore_values", np.full(16, .98), competitor=14)
        result = self.score(raw)
        self.assertFalse(result["validity_gates"]["guard_vs_value_oracle_full_probability"])
        self.assertEqual(float(np.max(np.abs(raw["guard_a0_b3/original_probability"] - raw["restore_values/original_probability"]))), 0)
        raw = copy.deepcopy(self.raw)
        self.set_probability(raw, "guard_restore_keys", np.full(16, .99), competitor=14)
        self.assertFalse(self.score(raw)["validity_gates"]["full_oracle_vs_native_full_probability"])

    def test_native_and_noop_failures_are_separate(self):
        raw = copy.deepcopy(self.raw)
        self.set_probability(raw, "native", np.full(16, .1))
        self.set_probability(raw, "guard_restore_keys", np.full(16, .1))
        self.assertFalse(self.score(raw)["validity_gates"]["native_accuracy"])
        raw = copy.deepcopy(self.raw)
        self.set_probability(raw, "grouped_noop", np.full(16, .41))
        self.assertFalse(self.score(raw)["validity_gates"]["grouped_noop_identity"])

    def test_each_residual_identity_and_tolerance_boundary(self):
        for metric, gate in (("l1_value_max_error_native", "correct_guard_l1_value_native"),
                             ("l1_query_max_error_native", "correct_guard_l1_query_native"),
                             ("l1_key_max_error_grouped", "correct_guard_l1_key_grouped")):
            raw = copy.deepcopy(self.raw)
            raw["guard_a0_b3/" + metric][0] = 1e-5
            self.assertTrue(self.score(raw)["validity_gates"][gate])
            raw["guard_a0_b3/" + metric][0] = np.nextafter(1e-5, np.inf)
            self.assertFalse(self.score(raw)["validity_gates"][gate])

    def test_one_query_failure_cannot_be_hidden_by_mean_accuracy(self):
        raw = copy.deepcopy(self.raw)
        probability = np.full(16, .98)
        probability[self.query == 2] = .1
        for name in ("guard_a0_b3", "restore_values"):
            self.set_probability(raw, name, probability.copy())
        result = self.score(raw)
        self.assertFalse(result["query_recovery_gates"]["2"])
        self.assertTrue(all(result["query_recovery_gates"][str(q)] for q in (0, 1, 3)))
        self.assertFalse(result["recovery_gates"]["each_query_accuracy"])

    def test_recovery_and_specificity_fail_independently(self):
        raw = copy.deepcopy(self.raw)
        for name in ("grouped_canonical", "grouped_noop"):
            self.set_probability(raw, name, np.full(16, .9))
        result = self.score(raw)
        self.assertFalse(result["recovery_gates"]["guard_minus_grouped_ci95_lower"])
        self.assertTrue(all(result["specificity_gates"].values()))
        raw = copy.deepcopy(self.raw)
        for name in self.wrong:
            self.set_probability(raw, name, np.full(16, .98))
        result = self.score(raw)
        self.assertTrue(all(result["recovery_gates"].values()))
        self.assertFalse(result["specificity_gates"]["guard_minus_mean_eight_ci95_lower"])

    def test_strict_ci_thresholds_reject_equality(self):
        for threshold in (.30, .05):
            self.assertFalse(lower_exceeds([threshold, 1], threshold))
            self.assertTrue(lower_exceeds([np.nextafter(threshold, np.inf), 1], threshold))

    def test_best_control_reselected_in_each_bootstrap(self):
        raw = copy.deepcopy(self.raw)
        for name in self.wrong:
            self.set_probability(raw, name, np.full(16, .6))
        oscillation = np.where((np.arange(16) // 4) % 2 == 0, .2, .9)
        self.set_probability(raw, self.wrong[0], oscillation)
        self.set_probability(raw, self.wrong[1], 1.1 - oscillation)
        result = self.score(raw)
        wrong = np.stack([raw[name + "/original_probability"] for name in self.wrong], axis=1)
        guard_draws = self.weights @ raw["guard_a0_b3/original_probability"]
        expected = ci(guard_draws - np.max(self.weights @ wrong, axis=1))
        np.testing.assert_allclose(result["comparisons"]["guard_minus_best_wrong"]["ci95_bootstrap"], expected)
        fixed_best = wrong.mean(0).argmax()
        fixed_interval = ci(guard_draws - self.weights @ wrong[:, fixed_best])
        self.assertLess(expected[0], fixed_interval[0])

    def test_mean_eight_and_factorial_are_paired_before_resampling(self):
        result = self.score(self.raw)
        correct = self.raw["guard_a0_b3/original_probability"]
        average = np.stack([self.raw[name + "/original_probability"] for name in self.wrong], axis=1).mean(1)
        np.testing.assert_allclose(result["comparisons"]["guard_minus_mean_eight"]["ci95_bootstrap"], ci(self.weights @ (correct - average)))
        interaction = self.raw["guard_restore_keys/original_probability"] - correct - self.raw["restore_keys/original_probability"] + self.raw["grouped_canonical/original_probability"]
        self.assertAlmostEqual(result["factorial_descriptive"]["interaction_full_minus_guard_minus_keys_plus_grouped"]["mean"], float(interaction.mean()))


if __name__ == "__main__":
    unittest.main()
