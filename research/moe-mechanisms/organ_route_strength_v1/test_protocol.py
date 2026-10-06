import copy
import unittest

import protocol as p


def invented_metadata():
    contexts = [{"id": f"s{s}-a{a}", "state": s, "action": a}
                for s in range(8) for a in range(3)]
    worlds = [{"id": f"dev{w:02d}", "transitions": [[(s + a + w) % 8 for s in range(8)]
                                                  for a in range(3)]} for w in range(4)]
    tail = {"contexts": copy.deepcopy(contexts), "worlds": worlds, "label_ids": list(range(34, 42))}
    schedule = []
    for context in contexts:
        for world in worlds:
            cell_id = f"{context['id']}-{world['id']}"
            for arm in ("B", "W"):
                i = len(schedule)
                schedule.append({"index": i, "id": f"{i:04d}", "arm": arm,
                                 "cell_id": cell_id, "context_id": context["id"], "world_id": world["id"]})
    return {"contexts": contexts, "schedule": schedule}, tail


def invented_plan():
    cells = p.metadata_cells(*invented_metadata())
    return {"cells": cells, "schedule": p.make_schedule(cells)}


class ProtocolTests(unittest.TestCase):
    def test_complete_fixed_schedule(self):
        plan = invented_plan()
        self.assertEqual(p.validate_schedule(plan["cells"], plan["schedule"]),
                         {"cells": 96, "suffixes": 768, "native": 288, "self_shams": 288, "fixed_B_routes": 192})
        self.assertEqual([(r["strength"], r["arm"]) for r in plan["schedule"][:8]], list(p.CONDITIONS))
        for row in plan["schedule"]:
            native = plan["schedule"][int(row["same_strength_native_id"])]
            self.assertEqual((native["cell_id"], native["strength"], native["arm"]),
                             (row["cell_id"], row["strength"], "N"))
            self.assertLessEqual(native["index"], row["index"])

    def test_source_missing_duplicate_reordered_rejected(self):
        for mutate in (lambda s: s.pop(), lambda s: s.__setitem__(1, copy.deepcopy(s[0])),
                       lambda s: s.reverse()):
            census, tail = invented_metadata()
            mutate(census["schedule"])
            with self.assertRaises(ValueError):
                p.metadata_cells(census, tail)

    def test_wrong_arm_and_context_rejected(self):
        census, tail = invented_metadata()
        census["schedule"][1]["arm"] = "B"
        with self.assertRaises(ValueError):
            p.metadata_cells(census, tail)
        census, tail = invented_metadata()
        census["contexts"][0]["state"] = False
        with self.assertRaises(ValueError):
            p.metadata_cells(census, tail)

    def test_invalid_world_target_and_labels_rejected(self):
        for mutation in (lambda t: t["worlds"][0]["transitions"][0].__setitem__(0, True),
                         lambda t: t["label_ids"].__setitem__(0, 999)):
            census, tail = invented_metadata()
            mutation(tail)
            with self.assertRaises(ValueError):
                p.metadata_cells(census, tail)

    def test_schedule_tampering_rejected(self):
        for mutate in (lambda s: s.pop(), lambda s: s.__setitem__(1, copy.deepcopy(s[0])),
                       lambda s: s[4].__setitem__("route_source", {"kind": "same_condition_native", "id": "0002"}),
                       lambda s: s[0].__setitem__("index", False),
                       lambda s: s[2].__setitem__("strength", "1/8")):
            plan = invented_plan()
            mutate(plan["schedule"])
            with self.assertRaises(ValueError):
                p.validate_schedule(plan["cells"], plan["schedule"])

    def test_every_cell_target_is_fixed_transition(self):
        plan = invented_plan()
        for cell in plan["cells"]:
            world = int(cell["world_id"][-2:])
            self.assertEqual(cell["target"], (cell["state"] + cell["action"] + world) % 8)
        for mutate in (lambda c: c.pop(), lambda c: c.__setitem__(1, copy.deepcopy(c[0]))):
            cells = copy.deepcopy(plan["cells"])
            mutate(cells)
            with self.assertRaises(ValueError):
                p.make_schedule(cells)


if __name__ == "__main__":
    unittest.main()
