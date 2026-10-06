import copy
import unittest

import metrics as m
from test_protocol import invented_plan


def invented_records(plan):
    j = {("0", "N"): -1.0, ("0", "S"): -1.0,
         ("1/16", "N"): 2.0, ("1/16", "S"): 2.0, ("1/16", "F0"): -1.0,
         ("1", "N"): 10.0, ("1", "S"): 10.0, ("1", "F0"): 6.0}
    rows = []
    for case in plan["schedule"]:
        scores = [0.0] * 8
        scores[case["target"]] = j[case["strength"], case["arm"]]
        rows.append({"id": case["id"], "scores": scores})
    return rows


class MetricTests(unittest.TestCase):
    def test_eight_label_math_and_fixed_target(self):
        row = m.score_eight([4, 1, 2, 3, 0, -1, -2, -3], 0)
        self.assertEqual(row["J"], 4)
        self.assertEqual(row["target_other"], {str(i): v for i, v in enumerate([0, 3, 2, 1, 4, 5, 6, 7]) if i})
        self.assertEqual(row["best_other_margin"], 1)
        self.assertTrue(row["correct"])
        self.assertEqual(row["choice"], 0)

    def test_exact_ties_are_unresolved_no_fallback(self):
        row = m.score_eight([1, 1, 0, 0, 0, 0, 0, 0], 0)
        self.assertEqual(row["winners"], [0, 1])
        self.assertIsNone(row["choice"])
        self.assertTrue(row["tie"])
        self.assertFalse(row["correct"])
        self.assertFalse(m.score_eight([0] * 8, 7)["correct"])
        self.assertFalse(m.score_eight([0, 1, 0, 0, 0, 0, 0, 0], 0)["correct"])

    def test_nonfinite_malformed_boolean_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                m.score_eight([bad] + [0] * 7, 0)
        for target in (False, -1, 8, 1.0):
            with self.assertRaises(ValueError):
                m.score_eight([0] * 8, target)
        with self.assertRaises(ValueError):
            m.score_eight([0] * 7, 0)

    def test_complete_contrasts_positive_helps(self):
        plan = invented_plan()
        report = m.evaluate_scores(plan, invented_records(plan))
        self.assertEqual(len(report["rows"]), 768)
        self.assertEqual(len(report["cells"]), 96)
        expected = {"route_1_16": 3, "route_1": 4, "route_full_minus_attenuated": 1,
                    "strength_full_minus_attenuated_native": 8,
                    "strength_full_minus_attenuated_fixed_B": 7}
        for key, value in expected.items():
            self.assertEqual(report["effects"][key], {"n": 96, "mean": value, "positive": 96, "negative": 0, "zero": 0})
        self.assertEqual(report["effects"]["sham_S_minus_N_0"]["zero"], 96)
        self.assertEqual(report["conditions"]["1/16:N"]["correct"], 96)
        self.assertEqual(report["conditions"]["1/16:F0"]["ties"], 96)
        self.assertFalse(report["implementation_admitted"])
        self.assertTrue(all(v["route_1"]["n"] == 24 for v in report["per_world_effects"].values()))

    def test_missing_duplicate_and_unknown_id_rejected(self):
        plan = invented_plan()
        for mutate in (lambda r: r.pop(), lambda r: r.__setitem__(1, copy.deepcopy(r[0])),
                       lambda r: r[-1].__setitem__("id", "9999")):
            records = invented_records(plan)
            mutate(records)
            with self.assertRaises(ValueError):
                m.evaluate_scores(plan, records)

    def test_final_row_nonfinite_stops_whole_report(self):
        plan = invented_plan()
        rows = invented_records(plan)
        rows[-1]["scores"][0] = float("nan")
        with self.assertRaises(ValueError):
            m.evaluate_scores(plan, rows)

    def test_shuffled_record_input_does_not_change_pairing(self):
        plan = invented_plan()
        records = invented_records(plan)
        self.assertEqual(m.evaluate_scores(plan, records), m.evaluate_scores(plan, list(reversed(records))))

    def test_shift_invariance_and_negative_zero_retained(self):
        first = m.score_eight([4, 1, 2, 3, 0, -1, -2, -3], 0)
        second = m.score_eight([v + 100 for v in first["scores"]], 0)
        self.assertEqual(first["J"], second["J"])
        self.assertEqual(first["target_other"], second["target_other"])
        self.assertEqual(m.summary([-1, 0, -0.0, 1]), {"n": 4, "mean": 0, "positive": 1, "negative": 1, "zero": 2})

    def test_finite_sham_difference_is_reported_not_admitted(self):
        plan = invented_plan()
        records = invented_records(plan)
        target = plan["schedule"][1]["target"]
        records[1]["scores"][target] += 1
        report = m.evaluate_scores(plan, records)
        self.assertEqual(report["effects"]["sham_S_minus_N_0"]["positive"], 1)
        self.assertFalse(report["implementation_admitted"])
        # Actual execution must fail exact raw-sham admission before this helper.


if __name__ == "__main__":
    unittest.main()
