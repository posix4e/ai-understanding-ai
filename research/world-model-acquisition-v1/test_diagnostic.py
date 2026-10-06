"""Invented finite arithmetic only; never reads pilot arrays or executes a model."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import diagnostic as d

spec = importlib.util.spec_from_file_location('separate_diagnostic_checker',
    Path(__file__).resolve().parents[2]/'work/world-model-acquisition-v1/diagnostic_check.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        states = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 1]]
        table = np.tile(np.arange(4)[:, None], (1, 4)); table[0, 1] = 3
        P = np.eye(4)[table]; P[0, 0] = [0, .5, .5, 0]
        pi = np.array([0, 1, 0, 0]); ties = np.ones(4, dtype=int)
        case = {'start': 0, 'goal': [1, 0], 'optimal_steps': 1}
        return states, table, P, pi, ties, case

    def test_blind_joint_cycle_and_internal_goal_continues(self):
        states, table, P, pi, ties, case = self.fixture()
        blind = d.simulate(table, states, P, pi, ties, case, True)
        self.assertEqual(blind['true_states'], [0, 0, 3])
        self.assertEqual(blind['internal_states'][:2], [0, 1])
        self.assertTrue(blind['success'])
        # True state revisits0 but internal state changes, including an imagined goal.
        feedback = d.simulate(table, states, P, pi, ties, case, False)
        self.assertEqual(feedback['stop_reason'], 'joint_cycle')
        self.assertEqual(feedback['steps'], 1)
        ref = c.execute(table.tolist(), states, P, pi.tolist(), ties.tolist(), case, True)
        c.compare(blind, ref)

    def test_true_goal_precedes_cycle_and_initial_goal(self):
        states, table, P, pi, ties, case = self.fixture()
        case['goal'] = [0, 0]
        r = d.simulate(table, states, P, pi, ties, case, True)
        self.assertTrue(r['success']); self.assertEqual(r['steps'], 0)
        c.compare(r, c.execute(table.tolist(), states, P, pi.tolist(), ties.tolist(), case, True))

    def test_blind_cap_and_lowest_index_successor_tie(self):
        states, table, P, pi, ties, case = self.fixture()
        r = d.simulate(table, states, P, pi, ties, case, True, cap=1)
        self.assertEqual(r['internal_states'], [0, 1]); self.assertEqual(r['stop_reason'], 'cap')
        c.compare(r, c.execute(table.tolist(), states, P, pi.tolist(), ties.tolist(), case, True, cap=1))

    def test_proper_losses_and_zero_count(self):
        P = np.array([[[1., 0.], [.25, .75], [.5, .5], [0., 1.]]])
        table = np.array([[0, 0, 1, 0]])
        r = d.loss_summary(P, table, [1, 2, 1, 1])
        self.assertEqual((r['n'], r['correct'], r['wrong'], r['ties']), (5, 1, 3, 1))
        self.assertEqual(r['zero_true_probability'], 1)
        self.assertAlmostEqual(r['mean_sum_brier'], (2*1.125+.5+2)/5)
        self.assertTrue(np.isfinite(r['capped_nll']))
        self.assertIsNone(d.loss_summary(P, table, [0, 0, 0, 0])['capped_nll'])

    def test_policy_independent_reductions_match_on_invented_kernel(self):
        states = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 1]]
        rng = np.random.default_rng(983)
        P = rng.uniform(.01, 1, (4, 4, 4)); P /= P.sum(axis=-1, keepdims=True)
        pi, ties = d.primary_policy(P, states, (1, 0))
        cp, ct = c.decision(P, states, (1, 0))
        self.assertEqual(pi.tolist(), cp); self.assertEqual(ties.tolist(), ct)
        hard = np.eye(4)[P.argmax(axis=-1)]
        pi, ties = d.primary_policy(hard, states, (1, 0))
        cp, ct = c.decision(hard, states, (1, 0))
        self.assertEqual(pi.tolist(), cp); self.assertEqual(ties.tolist(), ct)

    def test_paired_summary_including_failures(self):
        states, table, P, pi, ties, case = self.fixture()
        good = d.simulate(table, states, P, pi, ties, case, True)
        bad = d.simulate(table, states, P, pi, ties, case, False)
        b = [good, good]; r = [bad, good]
        s = d.summarize(r, b); c.compare(s, c.summarize(r, b))
        self.assertEqual(s['n'], 2); self.assertEqual(s['successes'], 1)
        self.assertEqual(s['paired_vs_soft_feedback']['success_to_failure'], 1)
        self.assertEqual(s['failed_case_indices'], [0])

    def test_checker_rejects_category_and_numeric_mutations(self):
        with self.assertRaises(ValueError): c.compare({'x': 1}, {'x': True})
        with self.assertRaises(ValueError): c.compare([.5], [.6])
        with self.assertRaises(ValueError): c.compare({'a': []}, {'b': []})


if __name__ == '__main__': unittest.main()
