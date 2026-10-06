"""Structural and information-boundary checks; no scientific fits."""
import copy
import inspect
import unittest
import numpy as np
from world import panel, evaluate
from selection import choose, key
from run import update_batches
from models import create, features

class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = panel()

    def test_panel_and_splits(self):
        self.assertEqual(len(self.maps), 12)
        self.assertEqual(self.maps, panel())
        self.assertEqual(len({m['layout_hash'] for m in self.maps}), 12)
        for i, m in enumerate(self.maps):
            self.assertEqual(m['phase'], 'engineering' if i < 4 else 'evaluation')
            splits = m['splits']
            self.assertEqual(set().union(*map(set, splits.values())), set(range(120)))
            self.assertEqual(sum(map(len, splits.values())), 120)
            for part, per_action in [('train', 18), ('query', 6), ('audit', 6)]:
                self.assertEqual([sum(r % 4 == a for r in splits[part]) for a in range(4)], [per_action]*4)

    def test_exact_planner(self):
        for m in self.maps:
            result, _ = evaluate(np.eye(30)[np.array(m['transitions'])], m)
            self.assertEqual(result['successes'], 600)
            self.assertEqual(result['mean_excess_steps_success'], 0)

    def test_hidden_targets_cannot_change_selection(self):
        m = self.maps[0]
        changed = copy.deepcopy(m)
        changed['transitions'] = [[0]*4 for _ in range(30)]
        changed['world']['key'] = [4, 4]
        P = np.random.default_rng(7).dirichlet(np.ones(30), size=(30, 4))
        for method in ('random', 'uncertainty', 'decision'):
            def select(data):
                return choose(P, data['states'], data['goals'], data['splits']['query'], method, data['id'], 11)
            self.assertEqual(select(m), select(changed))
        self.assertEqual(list(inspect.signature(choose).parameters), ['P', 'states', 'goals', 'remaining', 'method', 'map_id', 'seed'])

    def test_deterministic_predictions_have_zero_information(self):
        m = self.maps[0]
        P = np.eye(30)[np.array(m['transitions'])]
        for method in ('uncertainty', 'decision'):
            r = choose(P, m['states'], m['goals'], m['splits']['query'], method, m['id'], 11)
            self.assertTrue(all(abs(v) < 1e-12 for v in r['scores'].values()))
            expected = min(m['splits']['query'], key=lambda row: key('tie', m['id'], 11, row))
            self.assertEqual(r['selected'], expected)

    def test_entropy_and_no_repeats(self):
        m = self.maps[0]
        P = np.zeros((30, 4, 30)); P[:, :, 0] = 1
        candidates = m['splits']['query']
        s, a = divmod(candidates[0], 4)
        P[s, a] = 1/30
        self.assertEqual(choose(P, m['states'], m['goals'], candidates, 'uncertainty', m['id'], 11)['selected'], candidates[0])
        left = list(candidates)
        picked = []
        for _ in range(4):
            row = choose(P, m['states'], m['goals'], left, 'random', m['id'], 11)['selected']
            left.remove(row); picked.append(row)
        self.assertEqual(len(set(picked)), 4)

    def test_paired_batches_and_sealed_audit(self):
        m = self.maps[0]
        one = m['splits']['query'][:2]
        other = m['splits']['query'][2:4]
        b1 = update_batches(m['splits']['train'], one, 11, 2)
        b2 = update_batches(m['splits']['train'], other, 11, 2)
        replay = update_batches(m['splits']['train'], [], 11, 2, True)
        np.testing.assert_array_equal(b1[:, :16], b2[:, :16])
        np.testing.assert_array_equal(b1[:, :16], replay[:, :16])
        self.assertLessEqual(set(b1[:, 16:].flat), set(one))
        self.assertLessEqual(set(replay.flat), set(m['splits']['train']))
        self.assertFalse(set(b1.flat) & set(m['splits']['audit']))

    def test_parameterization_shapes(self):
        sf, sa = features(self.maps[0]['states'])
        for kind in ('direct', 'energy'):
            model, _ = create(kind, 11)
            self.assertEqual(tuple(model(sa[:2], sf).shape), (2, 30))

if __name__ == '__main__':
    unittest.main()
