"""Outcome-free synthetic fixtures for preregistered E3 scoring rules."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from experiment3.conditions import generate_conditions
from experiment3.score import ATTENTION_METRICS, FORECAST_METRICS, score_seed, stratified_weights


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conditions = generate_conditions()
        cls.by_id = {c["id"]: c for c in cls.conditions}
        cls.protocol = json.loads(Path("experiment3/protocol.json").read_text())
        cls.query = np.tile(np.arange(4), 4)
        cls.values = np.tile(np.arange(4), (16, 1))
        cls.arrays = {"query_pair": cls.query, "values": cls.values, "targets": cls.query.copy()}
        cls.weights = stratified_weights(cls.query, 128, 7200001)
        cls.raw = {}
        for condition in cls.conditions:
            slot = np.asarray(condition["slot_value_by_query"])[cls.query]
            probability = np.zeros((16, 16))
            probability[np.arange(16), slot] = 1
            cls.set_probability(cls.raw, condition, probability)
            for metric in ATTENTION_METRICS:
                cls.raw[condition["id"] + "/" + metric] = np.full((16, 4), 1 / 9)
        cls.forecasts = {metric: {c["id"]: float(cls.raw[c["id"] + "/" + metric].mean()) for c in cls.conditions}
                         for metric in FORECAST_METRICS}

    @classmethod
    def set_probability(cls, raw, condition, probability):
        name = condition["id"]
        slot = np.asarray(condition["slot_value_by_query"])[cls.query]
        predicted = probability.argmax(1)
        raw[name + "/class_probability"] = probability
        raw[name + "/original_probability"] = probability[np.arange(16), cls.query]
        raw[name + "/slot_probability"] = probability[np.arange(16), slot]
        raw[name + "/argmax_class"] = predicted
        raw[name + "/original_accuracy"] = (predicted == cls.query).astype(int)
        raw[name + "/slot_accuracy"] = (predicted == slot).astype(int)

    def score(self, raw):
        return score_seed(raw, self.arrays, self.conditions, self.forecasts, self.protocol, self.weights)[0]

    def test_strata_and_pairing_preserved(self):
        np.testing.assert_allclose(self.weights.sum(1), 1)
        for q in range(4):
            np.testing.assert_allclose(self.weights[:, self.query == q].sum(1), .25)
        np.testing.assert_array_equal(self.weights, stratified_weights(self.query, 128, 7200001))
        # Complementary condition outcomes must cancel within each row and draw.
        values = np.arange(16) / 16
        np.testing.assert_allclose((self.weights @ values) + (self.weights @ (1 - values)), 1)

    def test_success_fixture_and_forward_rule(self):
        result = self.score(self.raw)
        self.assertTrue(all(result["validity_gates"].values()))
        self.assertTrue(all(result["mechanism_gates"].values()))
        self.assertEqual(result["primary_probability_margin"]["mean"], 1)
        self.assertEqual(len(result["primary_conditions"]), 18)
        self.assertEqual(result["forecast_statistics"]["all"]["count"], 200)
        self.assertEqual(result["forecast_statistics"]["all"]["within_tolerance"], 200)
        secondary = result["secondary_forward_inverse"]
        self.assertEqual(len(secondary["conditions"]), 14)
        self.assertEqual(secondary["eligible_conditions_per_row"], [12])
        self.assertEqual(secondary["inverse_fidelity"]["mean"], 1)
        self.assertEqual(secondary["forward_fidelity"]["mean"], 0)
        self.assertEqual(secondary["inverse_minus_forward_probability"]["mean"], 1)
        self.assertEqual(len(result["secondary_probability_contrasts"]["coherent_minus_value_original_probability"]), 23)

    def test_one_failed_derangement_is_retained(self):
        raw = copy.deepcopy(self.raw)
        condition = next(c for c in self.conditions if c["primary_group"] == "value_derangement")
        probability = np.zeros((16, 16)); probability[np.arange(16), self.query] = 1
        self.set_probability(raw, condition, probability)
        result = self.score(raw)
        self.assertFalse(result["mechanism_gates"]["every_value_derangement"])
        self.assertFalse(result["primary_conditions"][condition["id"]]["pass"])
        self.assertTrue(result["mechanism_gates"]["every_coherent_control"])

    def test_one_failed_coherent_control_is_retained(self):
        raw = copy.deepcopy(self.raw)
        condition = next(c for c in self.conditions if c["primary_group"] == "coherent_control")
        probability = np.zeros((16, 16)); probability[:, 15] = 1
        self.set_probability(raw, condition, probability)
        result = self.score(raw)
        self.assertFalse(result["mechanism_gates"]["every_coherent_control"])
        self.assertFalse(result["primary_conditions"][condition["id"]]["pass"])

    def test_native_accuracy_failure(self):
        raw = copy.deepcopy(self.raw)
        probability = np.zeros((16, 16)); probability[:, 15] = 1
        for name in ("original_native", "original_noop"):
            self.set_probability(raw, self.by_id[name], probability.copy())
        result = self.score(raw)
        self.assertFalse(result["validity_gates"]["native_accuracy"])
        self.assertTrue(result["validity_gates"]["noop_identity"])

    def test_noop_error_failure(self):
        raw = copy.deepcopy(self.raw)
        probability = raw["original_noop/class_probability"].copy() * .9
        probability[:, 15] = .1
        self.set_probability(raw, self.by_id["original_noop"], probability)
        result = self.score(raw)
        self.assertFalse(result["validity_gates"]["noop_identity"])
        self.assertTrue(result["validity_gates"]["native_accuracy"])

    def test_margin_exactly_point_eight_does_not_pass(self):
        raw = copy.deepcopy(self.raw)
        for condition in self.conditions:
            if condition["primary_group"] == "value_derangement":
                probability = raw[condition["id"] + "/class_probability"].copy() * .8
                probability[:, 15] = .2
                self.set_probability(raw, condition, probability)
        result = self.score(raw)
        self.assertAlmostEqual(result["primary_probability_margin"]["mean"], .8)
        self.assertFalse(result["mechanism_gates"]["probability_margin_ci95_lower"])
        self.assertTrue(result["mechanism_gates"]["every_value_derangement"])

    def test_full_probabilities_audit_argmax_and_selected_targets(self):
        raw = copy.deepcopy(self.raw)
        raw["original_native/original_probability"] = np.full(16, .5)
        with self.assertRaisesRegex(ValueError, "full probabilities disagree"):
            self.score(raw)


if __name__ == "__main__":
    unittest.main()
