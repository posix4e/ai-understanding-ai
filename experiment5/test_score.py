"""Synthetic postprocessing fixtures; no model imports, inputs, or outcomes."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment5.conditions import GROUPED, generate_conditions
from experiment5.scorer import METRICS, STATE_METRICS, iter_saved_panel, primary_decisions, score_seed, stratified_weights
from experiment5.report import render


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = generate_conditions()
        grouped = next(l for l in full["primary"] if l["layout_indices"] == GROUPED)
        two = next(l for l in full["primary"] if sum(c["id"].startswith("sham_") for c in l["conditions"]) == 1)
        fixed_keys = next(l for l in full["boundary"] if l["layout_indices"][:4] == [0, 2, 4, 6])
        cls.grid = {"primary": [grouped, two], "boundary": [full["boundary"][0], fixed_keys],
                    "edge_panel": full["edge_panel"], "references": full["references"]}
        cls.protocol = json.loads(Path("experiment5/protocol.json").read_text())
        cls.forecasts = json.loads(Path("experiment5/predictions.json").read_text())["expected_aggregate"]["6"]
        cls.query = np.tile(np.arange(4), 8)
        cls.arrays = {"query_pair": cls.query, "targets": cls.query.copy(), "values": np.tile(np.arange(4), (len(cls.query), 1))}
        cls.weights = stratified_weights(cls.query, 100, 1001)
        cls.bweights = stratified_weights(cls.query[:16], 100, 1002)

    def fixture(self):
        n = len(self.query)
        panels = {}
        native_probability = self.probability(self.query, .985)
        for panel in ("references", "primary", "boundary", "edge_panel"):
            specs = self.grid[panel] if isinstance(self.grid[panel], list) else [self.grid[panel]]
            panels[panel] = {}
            for li, layout in enumerate(specs):
                raw = {}
                q = self.query[:16] if panel == "boundary" else self.query
                for c in layout["conditions"]:
                    name = c["id"]
                    probability = .98
                    if name in ("native", "oracle"):
                        probability = .985
                    elif name in ("unguarded", "noop"):
                        probability = .7
                    elif name.startswith("sham_"):
                        probability = .6 if li == 0 else .9
                    full = self.probability(q, probability)
                    if c.get("edge"):
                        i, j = c["edge"]
                        for row in np.flatnonzero(q == j):
                            full[row] = .05/14
                            full[row, j], full[row, i] = .70, .25
                    metrics = {metric: np.zeros(len(q)) for metric in STATE_METRICS}
                    metrics.update(self.endpoints(full, q, native_probability[:len(q)]))
                    raw[name] = metrics
                panels[panel][layout["id"]] = raw
        return panels

    @staticmethod
    def probability(targets, probability):
        full = np.full((len(targets), 16), (1-probability)/15, dtype=np.float64)
        full[np.arange(len(targets)), targets] = probability
        return full

    @staticmethod
    def endpoints(full, targets, native):
        return {"class_probability": full, "argmax_class": full.argmax(1), "accuracy": (full.argmax(1) == targets).astype(int),
                "original_probability": full[np.arange(len(targets)), targets],
                "class_probability_max_error_native": np.abs(full-native).max(1)}

    def score(self, panels):
        self.emitted = []
        return score_seed(lambda panel: iter(panels[panel].items()), self.arrays, self.grid, self.forecasts,
                          self.protocol, self.weights, self.bweights, lambda table, row: self.emitted.append((table, row)))

    def test_positive_fixture_all_gates_and_exact_nested_control_weighting(self):
        score, forecasts = self.score(self.fixture())
        self.assertTrue(all(score["validity_gates"].values()))
        self.assertTrue(all(score["primary_gates"].values()))
        self.assertTrue(all(score["secondary_gates"].values()))
        self.assertAlmostEqual(score["aggregate"]["primary_correct_minus_mean_shams_probability"], .98-(.6+.9)/2)
        self.assertAlmostEqual(score["aggregate"]["primary_native_minus_correct_probability"], .005)
        self.assertEqual(len(forecasts), 12)
        self.assertFalse(any(table == "primary_failures" for table, _ in self.emitted))

    def test_bootstrap_shared_strata_and_pairing(self):
        np.testing.assert_allclose(self.weights.sum(1), 1)
        for q in range(4):
            np.testing.assert_allclose(self.weights[:, self.query == q].sum(1), .25)
        np.testing.assert_array_equal(self.weights, stratified_weights(self.query, 100, 1001))
        x = np.arange(len(self.query), dtype=float)
        np.testing.assert_allclose(self.weights @ (x-x), 0)

    def test_exact_threshold_strictness(self):
        self.assertEqual(primary_decisions(.95, [-.1, .01], [.02, .1], self.protocol),
                         {"every_layout_query_accuracy": True, "native_minus_correct_upper": True, "correct_minus_mean_shams_lower": False})
        self.assertFalse(primary_decisions(.95-1e-8, [0, .01+1e-8], [.020001, .1], self.protocol)["every_layout_query_accuracy"])
        self.assertFalse(primary_decisions(1., [0, .01+1e-8], [.020001, .1], self.protocol)["native_minus_correct_upper"])

    def test_one_failed_layout_query_is_retained(self):
        panels = self.fixture()
        raw = panels["primary"][self.grid["primary"][0]["id"]]["correct"]
        full = raw["class_probability"].copy()
        full[0] = self.probability(np.array([1]), .98)[0]
        native = next(iter(panels["references"].values()))["native"]["class_probability"]
        raw.update(self.endpoints(full, self.query, native))
        score, _ = self.score(panels)
        self.assertFalse(score["primary_gates"]["every_layout_query_accuracy"])
        self.assertEqual(score["primary_failed_cells"], 1)
        self.assertEqual(sum(t == "primary_failures" for t, _ in self.emitted), 1)
        self.assertEqual(sum(t == "primary_failure_cases" for t, _ in self.emitted), 1)

    def test_noop_state_and_oracle_validity_failures(self):
        panels = self.fixture()
        primary = panels["primary"][self.grid["primary"][0]["id"]]
        primary["correct"]["l1_value_max_error_native"][0] = 2e-5
        primary["correct"]["l1_key_max_error_baseline"][0] = 2e-5
        primary["noop"]["l1_query_max_error_native"][0] = 2e-5
        native = next(iter(panels["references"].values()))["native"]["class_probability"]
        for name in ("noop", "oracle"):
            full = self.probability(self.query, .5)
            primary[name].update(self.endpoints(full, self.query, native))
        score, _ = self.score(panels)
        for name in ("correct_l1_value_native", "all_l1_key_baseline", "all_l1_query_native", "noop_full_probability", "oracle_full_probability"):
            self.assertFalse(score["validity_gates"][name])

    def test_invalid_full_probability_or_missing_panel_is_rejected(self):
        panels = self.fixture()
        raw = next(iter(panels["primary"].values()))["correct"]
        raw["class_probability"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.score(panels)

    def test_native_and_boundary_identity_failure(self):
        panels = self.fixture()
        ref = next(iter(panels["references"].values()))
        full = self.probability((self.query+1)%4, .985)
        ref["native"].update(self.endpoints(full, self.query, full))
        for panel, layouts in panels.items():
            for raw in layouts.values():
                for metrics in raw.values():
                    length = len(metrics["accuracy"])
                    metrics["class_probability_max_error_native"] = np.abs(metrics["class_probability"]-full[:length]).max(1)
        next(iter(panels["boundary"].values()))["own_key_self"]["l1_value_max_error_own_key_self"][0] = 2e-5
        score, _ = self.score(panels)
        self.assertFalse(score["validity_gates"]["native_accuracy"])
        self.assertFalse(score["validity_gates"]["own_key_self_l1_value_reference"])
        panels = self.fixture()
        panels["boundary"].pop(next(iter(panels["boundary"])))
        with self.assertRaises(ValueError):
            self.score(panels)

    def test_edge_uses_only_source_query_and_equal_edges(self):
        panels = self.fixture()
        before, _ = self.score(panels)
        raw = next(iter(panels["edge_panel"].values()))
        native = next(iter(panels["references"].values()))["native"]["class_probability"]
        for c in self.grid["edge_panel"]["conditions"]:
            name, (_, source) = c["id"], c["edge"]
            full = raw[name]["class_probability"].copy()
            full[self.query != source] = self.probability((self.query[self.query != source]+1)%4, .9)
            raw[name].update(self.endpoints(full, self.query, native))
        after, _ = self.score(panels)
        self.assertEqual(before["edge"], after["edge"])
        self.assertAlmostEqual(before["edge"]["mean"], np.mean([r["mean"] for r in before["per_edge"].values()]))

    def test_numpy_shard_reader_preserves_names_and_detects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "confirmatory/seed_6/primary"
            folder.mkdir(parents=True)
            np.savez(folder / "shard_00000.npz", **{"layout_01234567/correct/accuracy": np.ones(3)})
            def ledger():
                entries = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "layout_ids": ["layout_01234567"]}
                           for p in sorted(folder.glob("*.npz"))]
                (folder.parent / "index.json").write_text(json.dumps({"seed": 6, "panels": {"primary": {"shards": entries}}}))
            ledger()
            rows = list(iter_saved_panel(6, "primary", root))
            np.testing.assert_array_equal(rows[0][1]["correct"]["accuracy"], np.ones(3))
            np.savez(folder / "shard_00001.npz", **{"layout_01234567/correct/accuracy": np.ones(3)})
            ledger()
            with self.assertRaises(ValueError):
                list(iter_saved_panel(6, "primary", root))

    def test_report_preserves_primary_failure_and_separate_secondary_success(self):
        score, _ = self.score(self.fixture())
        scores = {str(seed): copy.deepcopy(score) for seed in self.protocol["model_seeds"]}
        scores["6"]["primary_gates"]["every_layout_query_accuracy"] = False
        scores["6"]["primary_failed_cells"] = 1
        decisions = {"all_six_valid": True, "all_six_recovery": False, "all_six_specificity": True,
                     "primary_claim_pass": False, "secondary": {"boundary_logical_prefix": True, "boundary_own_key_self": True, "edge_redirection": True}}
        numeric = {"count": 72, "mae": .01, "rmse": .02, "within_tolerance": 70}
        text = render(scores, decisions, numeric, self.protocol, {"commit": "synthetic"},
                      {"preregistration_commit": "synthetic", "utc": "fixture"}, {"utc": "fixture"})
        self.assertIn("primary claim did not pass", text)
        self.assertIn("every_layout_query_accuracy", text)
        self.assertIn("23 boundary value orders plus", text)
        self.assertIn("primary_failure_cases.csv", text)


if __name__ == "__main__":
    unittest.main()
