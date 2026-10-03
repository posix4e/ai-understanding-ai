"""Synthetic-only checks of scoring, decision boundaries and saved labels."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiment6.score import (CONDITIONS, decisions, interval, main, render_markdown,
                               score_conditions, stratified_weights, validate_input_labels, validate_raw)


def fixture(n=16):
    candidates = np.arange(100, 116)
    result = {}
    for name in CONDITIONS:
        correct = name in ("native", "guard_block0", "guard_block5", "guard_all")
        probabilities = np.full((n, 16), .001, dtype=np.float32)
        probabilities[:, 0 if correct else 1] = .5
        result[name] = {
            "candidate_probs": probabilities,
            "top_token_ids": np.full(n, 100 if correct else 101),
            "target_token_ids": np.full(n, 100),
            "target_indices": np.zeros(n, dtype=int),
            "query_pair": np.arange(n) % 4,
        }
    return result, candidates


def gate_values():
    return {"native_accuracy": .8, "native_strata": [.7]*4,
            "damage": {"mean": .10, "ci95": [.01, .20]},
            "repair": {"mean": .2, "ci95": [.051, .3]},
            "deficit": {"mean": .01, "ci95": [-.01, .05]},
            "specificity": {"mean": .05, "ci95": [.021, .10]}, "valid": True}


class ScoringTests(unittest.TestCase):
    def test_inclusive_and_strict_gate_boundaries(self):
        self.assertTrue(decisions(**gate_values())["all_pass"])
        alterations = [("native_accuracy", np.nextafter(.8, 0)), ("native_strata", [.7, .7, .7, np.nextafter(.7, 0)]),
                       ("damage", {"mean": np.nextafter(.1, 0), "ci95": [.01, .2]}),
                       ("damage", {"mean": .1, "ci95": [0, .2]}),
                       ("repair", {"mean": .2, "ci95": [.05, .3]}),
                       ("deficit", {"mean": .01, "ci95": [-.01, np.nextafter(.05, 1)]}),
                       ("specificity", {"mean": .05, "ci95": [.02, .1]}), ("valid", False)]
        for key, value in alterations:
            with self.subTest(key=key, value=value):
                arguments = gate_values()
                arguments[key] = value
                self.assertFalse(decisions(**arguments)["all_pass"])

    def test_classification_does_not_hide_other_failures(self):
        arguments = gate_values()
        arguments.update(native_accuracy=.79, repair={"mean": 0, "ci95": [-.1, .1]})
        result = decisions(**arguments)
        self.assertEqual(result["classification"], "task_invalid")
        self.assertEqual(len(result["failed_gates"]), 2)
        arguments = gate_values()
        arguments["damage"]["mean"] = .09
        self.assertEqual(decisions(**arguments)["classification"], "no_damage")
        arguments = gate_values()
        arguments["specificity"]["ci95"][0] = .02
        self.assertEqual(decisions(**arguments)["classification"], "repair_failed")

    def test_bootstrap_is_paired_and_preserves_stratum_weights(self):
        query = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        weights = stratified_weights(query, draws=100, seed=123)
        np.testing.assert_allclose(weights.sum(1), 1)
        for q in range(4):
            np.testing.assert_allclose(weights[:, query == q].sum(1), .25)
        # Stratum-constant differences remain exactly 1.5 in every bootstrap draw.
        self.assertEqual(interval(query.astype(float), weights), {"mean": 1.5, "ci95": [1.5, 1.5]})
        self.assertEqual(interval(query-query, weights)["ci95"], [0., 0.])
        np.testing.assert_array_equal(weights, stratified_weights(query, draws=100, seed=123))

    def test_synthetic_all_pass_and_raw_vs_conditional_probability(self):
        raw, candidates = fixture()
        scores = score_conditions(raw, {"all_pass": True, "checks": {"noop": True}}, candidates, 16)
        self.assertTrue(scores["all_pass"])
        native = scores["conditions"]["native"]
        self.assertAlmostEqual(native["raw_target_probability"]["mean"], .5)
        self.assertAlmostEqual(native["conditional_target_probability"]["mean"], .5/.515, places=7)
        self.assertEqual(scores["contrasts"]["repair_guard_minus_grouped_accuracy"]["ci95"], [1, 1])
        self.assertIn("Classification: all_pass", render_markdown(scores))
        # Candidate ranking stays perfect even when the unrestricted winner is not a candidate.
        for arrays in raw.values():
            arrays["top_token_ids"][:] = 999
        scores = score_conditions(raw, {"all_pass": True, "checks": {"noop": True}}, candidates)
        self.assertEqual(scores["conditions"]["native"]["unrestricted_accuracy"]["mean"], 0)
        self.assertEqual(scores["conditions"]["native"]["restricted_accuracy"]["mean"], 1)

    def test_secondary_success_cannot_rescue_primary_failure(self):
        raw, candidates = fixture()
        raw["guard_block0"] = copy.deepcopy(raw["grouped_canonical"])
        scores = score_conditions(raw, {"all_pass": True, "checks": {"noop": True}}, candidates)
        self.assertEqual(scores["classification"], "repair_failed")
        self.assertEqual(scores["conditions"]["guard_all"]["restricted_accuracy"]["mean"], 1)
        self.assertIn("Classification: repair_failed", render_markdown(scores))

    def test_mean_control_is_per_case_average_not_selected_control(self):
        raw, candidates = fixture()
        raw["sham_0"] = copy.deepcopy(raw["guard_block0"])
        scores = score_conditions(raw, {"all_pass": True, "checks": {"noop": True}}, candidates)
        contrast = scores["contrasts"]["specificity_guard_minus_mean_shams_conditional_probability"]
        expected = 7/8 * ((.5-.001)/.515)
        self.assertAlmostEqual(contrast["mean"], expected, places=7)
        best = scores["contrasts"]["guard_minus_best_point_sham_conditional_probability_descriptive"]
        self.assertEqual(best["sham"], "sham_0")
        self.assertEqual(best["mean"], 0.)

    def test_reject_malformed_arrays_or_metadata(self):
        corruptions = [lambda r: r.pop("sham_7"),
                       lambda r: r["native"]["candidate_probs"].__setitem__((0, 0), np.nan),
                       lambda r: r["native"]["candidate_probs"].__setitem__((0, slice(None)), 0),
                       lambda r: r["native"]["candidate_probs"].__setitem__((0, slice(None)), .1),
                       lambda r: r["sham_0"]["query_pair"].__setitem__(0, 3),
                       lambda r: r["sham_0"]["target_token_ids"].__setitem__(0, 101),
                       lambda r: r["sham_0"]["top_token_ids"].__setitem__(0, 100)]
        for corrupt in corruptions:
            raw, candidates = fixture()
            corrupt(raw)
            with self.assertRaises(ValueError):
                validate_raw(raw, candidates)

    def test_frozen_inputs_and_validity_record_are_checked(self):
        raw, candidates = fixture()
        cases = [{"values": [20]*4, "query_pair": i % 4} for i in range(16)]
        validate_input_labels(raw, cases, list(range(20, 36)))
        cases[0]["values"][0] = 21
        with self.assertRaises(ValueError):
            validate_input_labels(raw, cases, list(range(20, 36)))
        with self.assertRaises(ValueError):
            score_conditions(raw, {"all_pass": True, "checks": {"noop": False}}, candidates)
        scores = score_conditions(raw, {"all_pass": False, "checks": {"noop": False}}, candidates)
        self.assertEqual(scores["classification"], "implementation_invalid")

    def test_cli_saved_array_to_json_and_report_contract(self):
        raw, candidates = fixture(256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "confirmatory"
            out.mkdir()
            for name, arrays in raw.items():
                np.savez(out / f"{name}.npz", **arrays)
            (out / "validity.json").write_text(json.dumps({"all_pass": True, "checks": {"noop": True}}))
            cases = [{"values": [20]*4, "query_pair": i % 4} for i in range(256)]
            inputs = root / "inputs.json"
            inputs.write_text(json.dumps(cases))
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"n": 256, "candidate_token_ids": candidates.tolist(),
                                            "inputs_path": str(inputs), "values": list(range(20, 36))}))
            with patch("sys.argv", ["score", "--root", str(root), "--protocol", str(protocol)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((out / "scores.json").read_text())
            self.assertEqual(result["classification"], "all_pass")
            self.assertEqual(set(result["raw_sha256"]), set(CONDITIONS))
            self.assertEqual(len(result["inputs_sha256"]), 64)
            self.assertIn("Restricted accuracy ranks 16", (root / "RESULTS.md").read_text())


if __name__ == "__main__":
    unittest.main()
