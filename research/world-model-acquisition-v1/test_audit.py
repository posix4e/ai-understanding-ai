"""Invented arithmetic/archive fixtures; no training, torch or study outputs."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("independent_acquisition_audit", Path(__file__).with_name("audit.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, allow_nan=False))


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = a.reference_maps()

    def test_fixed_geometry_and_disjoint_splits(self):
        self.assertEqual(len(self.maps), 12)
        self.assertEqual(sum(m["phase"] == "engineering" for m in self.maps), 4)
        for i, m in enumerate(self.maps):
            self.assertEqual(m["id"], f"map_{i:02d}")
            self.assertEqual(len(m["states"]), 30)
            self.assertEqual(len(m["goals"]), 21)
            all_ids = sum(m["splits"].values(), [])
            self.assertEqual(sorted(all_ids), list(range(120)))
            for name, per_action in (("train", 18), ("query", 6), ("audit", 6)):
                self.assertEqual([sum(r % 4 == j for r in m["splits"][name]) for j in range(4)], [per_action]*4)
            self.assertEqual(m["world"]["start"], [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]][i % 4])

    def test_transition_key_door_and_walls(self):
        w = {"walls": [[2, 0]], "door": [2, 1], "key": [0, 1]}
        self.assertEqual(a.transition((0, 0, 0), 2, w), (0, 1, 1))
        self.assertEqual(a.transition((0, 1, 1), 0, w), (0, 0, 1))
        self.assertEqual(a.transition((1, 1, 0), 1, w), (1, 1, 0))
        self.assertEqual(a.transition((1, 1, 1), 1, w), (2, 1, 1))
        self.assertEqual(a.transition((1, 0, 1), 1, w), (1, 0, 1))

    def test_all_oracle_graphs_reach_at_shortest_length(self):
        for m in self.maps:
            P = np.eye(30)[np.asarray(m["transitions"])]
            summary, rows = a.navigation(P, m)
            self.assertEqual(summary["successes"], 600)
            self.assertEqual(summary["mean_excess_steps_success"], 0)
            self.assertTrue(all(r["steps"] == r["optimal_steps"] for r in rows))

    def test_finite_softmax_zero_mass_and_ties(self):
        logits = np.zeros((30, 4, 30), dtype=np.float32)
        P = a.probabilities(logits)
        np.testing.assert_array_equal(P, np.full_like(P, 1/30))
        self.assertEqual(a.prediction_metrics(P, self.maps[0])["correct"], 0)
        logits[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            a.probabilities(logits)
        P = np.zeros_like(P)
        P[:, :, 0] = 1
        metric = a.prediction_metrics(P, self.maps[0])
        self.assertGreater(metric["zero_true_probability"], 0)
        self.assertTrue(np.isfinite(metric["capped_nll"]))

    def test_unique_entropy_max_and_zero_entropy(self):
        P = np.zeros((30, 4, 30))
        P[:, :, 0] = 1
        P[0, 1] = 1/30
        m = self.maps[0]
        row = a.selector(P, m["states"], m["goals"], [0, 1, 2], "uncertainty", m["id"], 11)
        self.assertEqual(row["selected"], 1)
        self.assertAlmostEqual(row["scores"]["1"], np.log(30))
        self.assertEqual(row["scores"]["0"], 0)

    def test_random_identity_and_remaining_pool(self):
        m = self.maps[0]
        P = np.full((30, 4, 30), 1/30)
        pool = m["splits"]["query"]
        wanted = min(pool, key=lambda r: hashlib.sha256(f"wm-active-v1:random:{m['id']}:11:{r}".encode()).hexdigest())
        result = a.selector(P, m["states"], m["goals"], pool, "random", m["id"], 11)
        self.assertEqual(result["selected"], wanted)
        self.assertEqual(set(result["scores"]), set(map(str, pool)))
        for bad in ([], [1, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                a.selector(P, m["states"], m["goals"], bad, "random", m["id"], 11)

    def test_hand_information_value_and_confident_wrong_blindness(self):
        # Revelation reveals a cost 1 or 5 equally; safe alternative costs 3.
        # Before =3, after = (1+3)/2 =2, hence gain1.
        self.assertEqual(a.local_information_value(np.array([.5, .5]), np.array([3., 3., 6., 7.]), np.array([0., 4.]), 0), 1)
        self.assertEqual(a.local_information_value(np.array([1., 0.]), np.array([1., 3., 6., 7.]), np.array([0., 4.]), 0), 0)
        m = self.maps[0]
        P = np.zeros((30, 4, 30))
        P[:, :, 0] = 1  # Arbitrary confidently wrong model, no oracle consulted.
        report = a.selector(P, m["states"], m["goals"], m["splits"]["query"], "decision", m["id"], 11)
        self.assertTrue(all(v == 0 for v in report["scores"].values()))
        tied = sorted(m["splits"]["query"], key=lambda r: a.h(f"wm-active-v1:tie:{m['id']}:11:{r}"))
        self.assertEqual(report["tied"], tied)

    def test_jensen_nonnegative_independent_enumeration(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            weights = rng.dirichlet(np.ones(5))
            continuation = rng.random(5)*10
            own = 1+sum(float(x*y) for x, y in zip(weights, continuation))
            q = [own, 2., 4., 8.]
            expected = min(q)-sum(float(p*min(2, 1+v)) for p, v in zip(weights, continuation))
            actual = a.local_information_value(weights, q, continuation, 0)
            self.assertAlmostEqual(actual, expected)
            self.assertGreaterEqual(actual, -1e-12)

    def test_updates_share_original_stream_without_unrevealed_targets(self):
        train = self.maps[0]["splits"]["train"]
        left = a.expected_update_batches(train, [3], 11, 1)
        right = a.expected_update_batches(train, [4], 11, 1)
        replay = a.expected_update_batches(train, [], 11, 1, True)
        np.testing.assert_array_equal(left[:, :16], right[:, :16])
        np.testing.assert_array_equal(left[:, :16], replay[:, :16])
        self.assertTrue((left[:, 16:] == 3).all())
        self.assertTrue(set(replay.reshape(-1)).issubset(train))

    def test_true_state_cycle_is_retained_as_failure(self):
        m = self.maps[0]
        P = np.eye(30)[:, None, :].repeat(4, axis=1)
        summary, rows = a.navigation(P, m)
        self.assertGreater(summary["cycles"], 0)
        self.assertEqual(len(rows), 600)
        self.assertTrue(all(r["success"] or r["cycle"] or r["steps"] == 40 for r in rows))


class ArchiveTests(unittest.TestCase):
    def test_manifest_missing_corrupt_or_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x"
            p.write_bytes(b"invented")
            a.validate_manifest(d, {"x": a.digest(p)}, {"x"})
            for mapping in ({}, {"x": "0"*64}, {"../x": "0"*64}):
                with self.assertRaises(ValueError):
                    a.validate_manifest(d, mapping, {"x"})

    def test_training_target_hash_and_counts(self):
        batches = np.zeros((250, 32), dtype=np.int64)
        targets = np.arange(120, dtype=np.int64)
        record = {"updates": 250, "loss_trace": [{"update": 250, "loss": 1.}],
                  "used_targets_sha256": hashlib.sha256(targets[batches].tobytes()).hexdigest(),
                  "seconds": .1, "round": 1, "queries": 1}
        a.validate_training(record, batches, targets, 250, {"round": 1, "queries": 1})
        for key, bad in (("used_targets_sha256", "0"*64), ("updates", True), ("queries", 2)):
            altered = dict(record, **{key: bad})
            with self.assertRaises(ValueError):
                a.validate_training(altered, batches, targets, 250, {"round": 1, "queries": 1})

    def test_stop_is_not_graded(self):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d)/"engineering"/"STOP.json", {"reason": "invented"})
            with patch.object(a, "validate_plan", side_effect=AssertionError("must not read/grade")):
                with self.assertRaisesRegex(ValueError, "STOP remains STOP"):
                    a.audit(d, "engineering")

    def test_complete_invented_acquisition_and_episode_loader(self):
        # One map/model, four arms and all rounds exercises the real archive path.
        # This is a serializer fixture, not optimization evidence.
        with tempfile.TemporaryDirectory() as d, patch.object(a, "SEEDS", (11,)), patch.object(a, "KINDS", ("direct",)):
            directory = Path(d)
            m = a.reference_maps()[0]
            folder = directory/m["id"]/"direct_11"
            folder.mkdir(parents=True)
            logits = np.zeros((30, 4, 30), dtype=np.float32)
            logits[:, :, 0] = 90  # Finite deterministic-near forecast, invented.
            target = np.asarray(m["transitions"], dtype=np.int64).reshape(120)
            def training(batches, extra):
                return {"updates": len(batches), "loss_trace": [{"update": i, "loss": 1.} for i in range(250, len(batches)+1, 250)],
                        "used_targets_sha256": hashlib.sha256(target[batches].tobytes()).hexdigest(), "seconds": .01, **extra}
            batches = np.random.default_rng(11).choice(m["splits"]["train"], (1500, 32))
            np.save(folder/"base_batches.npy", batches)
            np.save(folder/"base.npy", logits)
            (folder/"base.pt").write_bytes(b"opaque invented checkpoint")
            initial = training(batches, {"map": m["id"], "model": "direct_11"})
            write(folder/"base_training.json", initial)
            chains = []
            for arm in a.ARMS:
                (folder/arm).mkdir()
                acquired = []
                previous = folder/"base.npy"
                for rnd in range(1, 5):
                    remaining = sorted(set(m["splits"]["query"])-set(acquired))
                    if arm == "replay":
                        choice = {"selected": None, "revealed_target": None, "scores": {}, "tied": [], "clamped_negative_gains": 0, "minimum_raw_gain": 0.}
                    else:
                        choice = a.selector(a.probabilities(logits), m["states"], m["goals"], remaining, arm, m["id"], 11)
                        acquired.append(choice["selected"])
                        choice["revealed_target"] = int(target[choice["selected"]])
                    choice.update(round=rnd, method=arm, seed=11, map=m["id"], remaining_before=remaining,
                                  acquired=list(acquired), prediction_sha256=a.digest(previous), seconds=.01)
                    stem = folder/arm/f"round_{rnd}"
                    write(str(stem)+"_query.json", choice)
                    batches = a.expected_update_batches(m["splits"]["train"], acquired, 11, rnd, arm == "replay")
                    np.save(str(stem)+"_batches.npy", batches)
                    np.save(str(stem)+".npy", logits)
                    Path(str(stem)+".pt").write_bytes(b"opaque invented checkpoint")
                    write(str(stem)+"_training.json", training(batches, {"round": rnd, "queries": len(acquired)}))
                    previous = Path(str(stem)+".npy")
                chains.append({"map": m["id"], "model": "direct_11", "arm": arm, "queries": acquired})
            write(directory/"INITIAL_TRAINING_COMPLETE.json", {"at_utc": "invented", "fits": [initial]})
            write(directory/"ACQUISITION_COMPLETE.json", {"at_utc": "invented", "chains": chains, "files_sha256": {}})
            predictions, actual_chains = a.audit_acquisition(directory, [m])
            self.assertEqual(len(predictions), 17)
            self.assertEqual(chains, actual_chains)
            report = {"map": m["id"], "phase": m["phase"], "models": {}}
            with gzip.open(directory/m["id"]/"episodes.jsonl.gz", "wt") as stream:
                def grade(P, name, arm, budget):
                    summary, rows = a.navigation(P, m)
                    for row in rows:
                        stream.write(json.dumps({"model": name, "arm": arm, "budget": budget, **row})+"\n")
                    return {"planning": summary, "prediction_audit": a.prediction_metrics(P, m)}
                report["oracle"] = grade(np.eye(30)[np.asarray(m["transitions"])], "oracle", "oracle", 0)
                report["models"]["direct_11"] = {"base": grade(predictions[(m["id"], "direct_11", "base", 0)], "direct_11", "base", 0), "arms": {}}
                for arm in a.ARMS:
                    report["models"]["direct_11"]["arms"][arm] = {str(b): grade(predictions[(m["id"], "direct_11", arm, b)], "direct_11", arm, b) for b in (1, 2, 4)}
            report["primary_difference"] = 0.
            sets = {r["arm"]: set(r["queries"]) for r in chains}
            report["query_overlap"] = [{"model": "direct_11", "decision_random": len(sets["decision"] & sets["random"]),
                                       "decision_uncertainty": len(sets["decision"] & sets["uncertainty"]), "uncertainty_random": len(sets["uncertainty"] & sets["random"])}]
            write(directory/m["id"]/"report.json", report)
            write(directory/"report.json", {"status": "COLLECTED_PENDING_AUDIT", "phase": "engineering", "maps": [report],
                  "primary_mean_difference": 0., "positive_maps": 0, "seconds": .1, "peak_rss_bytes": 1, "new_provider_calls": 0, "new_estimated_usd": 0})
            _, count = a.audit_metrics(directory, [m], predictions, chains)
            self.assertEqual(count, 8400)
            # A changed selected target is structural, even if quality looks fine.
            path = folder/"decision"/"round_4_query.json"
            bad = a.read(path)
            bad["revealed_target"] = (bad["revealed_target"]+1) % 30
            write(path, bad)
            with self.assertRaises(ValueError):
                a.audit_acquisition(directory, [m])


if __name__ == "__main__":
    unittest.main()
