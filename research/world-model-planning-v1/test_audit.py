"""Invented arithmetic/rule fixtures only; no fitted model or saved study data."""
import math
import unittest

import numpy as np

import audit as a


class IndependentAuditTests(unittest.TestCase):
    def test_literal_boundary_key_and_door_rules(self):
        self.assertEqual(a.transition((0, 0, 0), 0), (0, 0, 0))
        self.assertEqual(a.transition((1, 1, 0), 1), (1, 1, 0))
        self.assertEqual(a.transition((1, 2, 0), 1), (1, 2, 0))
        self.assertEqual(a.transition((1, 2, 1), 1), (2, 2, 1))
        self.assertEqual(a.transition((0, 3, 0), 2), (0, 4, 1))
        self.assertEqual(a.transition((1, 4, 0), 3), (0, 4, 1))
        self.assertEqual(a.transition((0, 4, 1), 0), (0, 3, 1))
        with self.assertRaises(ValueError):
            a.transition((0, 0, 0), True)

    def test_reverse_bfs_and_exact_planning_on_three_state_fixture(self):
        states = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        table = np.array([[0, 1, 0, 0], [1, 2, 1, 0], [2, 2, 2, 1]])
        p = np.eye(3)[table]
        self.assertEqual(a.goal_distances(table, states, (2, 0)), [2, 1, 0])
        actions, ties, q = a.bellman_policy(p, states, (2, 0), 1)
        self.assertEqual(actions[:2], [1, 1])
        ep = a.replay_episode(table, states, 0, (2, 0), actions, ties)
        self.assertEqual(ep["states"], [0, 1, 2])
        self.assertTrue(ep["success"])
        self.assertFalse(ep["cycle"])
        # Goal beyond H=1 is still reached under receding-horizon control.
        self.assertEqual(ep["steps"], 2)

    def test_ties_and_cycle_are_explicit_not_global_planning_failure(self):
        states = [(0, 0, 0), (1, 0, 0)]
        table = np.array([[0, 1, 0, 0], [1, 1, 1, 0]])
        p = np.full((2, 4, 2), 0.5)
        actions, ties, _ = a.bellman_policy(p, states, (1, 0), 3)
        self.assertEqual(actions, [0, 0])
        self.assertEqual(ties, [4, 4])
        ep = a.replay_episode(table, states, 0, (1, 0), actions, ties)
        self.assertEqual(ep["states"], [0, 0])
        self.assertEqual(ep["tie_counts"], [4])
        self.assertTrue(ep["cycle"])
        self.assertFalse(ep["success"])

    def test_softmax_sign_support_and_bad_arrays(self):
        x = np.zeros((30, 4, 30), dtype=np.float32)
        x[:, :, 4] = 2
        p = a.probability_from_logits(x)
        self.assertTrue(np.all(p.argmax(-1) == 4))
        self.assertTrue(np.allclose(p.sum(-1), 1, rtol=0, atol=1e-14))
        for bad in (x.astype(np.float64), x[:2], np.full_like(x, np.nan)):
            with self.assertRaises(ValueError):
                a.probability_from_logits(bad)

    def test_zero_nll_floor_disclosed_and_strict_ties(self):
        p = np.array([[1., 0.], [.5, .5], [0., 1.]])
        report = a.categorical_metrics(p, [1, 0, 1])
        self.assertEqual((report["correct"], report["ties"], report["zero_true_probability"]), (1, 1, 1))
        expected = (-math.log(np.finfo(np.float64).tiny) + math.log(2)) / 3
        self.assertAlmostEqual(report["capped_nll"], expected)
        self.assertEqual(report["true_probability"], .5)

    def test_open_loop_full_probability_propagation(self):
        table = np.array([[0, 1, 0, 0], [1, 1, 1, 0]])
        p = np.eye(2)[table]
        case = {"start": 0, "actions": [1, 3] * 5}
        report = a.rollout_metrics(p, table, [case])
        for value in report.values():
            self.assertEqual(value["accuracy"], 1)
            self.assertEqual(value["true_probability"], 1)
            self.assertEqual(value["capped_nll"], 0)

    def test_strict_counts_coverage_and_float_tolerance(self):
        a.compare_numeric({"n": 1, "p": .5}, {"n": 1, "p": .5 + 1e-12})
        for observed in ({"n": True, "p": .5}, {"n": 1}, {"n": 1, "p": .6}):
            with self.assertRaises(ValueError):
                a.compare_numeric(observed, {"n": 1, "p": .5})

    def test_success_summary_retains_failed_denominator(self):
        rows = [{"success": True, "cycle": False, "steps": 4, "optimal_steps": 2, "tie_counts": [1, 1, 2, 1]},
                {"success": False, "cycle": True, "steps": 1, "optimal_steps": 7, "tie_counts": [4]}]
        report = a.planning_summary(rows)
        self.assertEqual(report["success_rate"], .5)
        self.assertEqual(report["mean_excess_steps_success"], 2)
        self.assertEqual(report["cycles"], 1)
        self.assertEqual(report["decisions"], 5)
        self.assertEqual(report["tied_decisions"], 2)
        self.assertEqual(report["by_distance"]["7+"], {"n": 1, "successes": 0})


if __name__ == "__main__":
    unittest.main()
