"""Independent saved-data verifier; no production, torch or training imports.

The decision selector is frozen-value exact-row-revelation sensitivity, not the
expected improvement from an SGD update. Optimization itself is not replayed.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np

SEEDS = (11, 29, 47)
KINDS = ("direct", "energy")
ARMS = ("random", "uncertainty", "decision", "replay")
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def same(a, b):
    return canonical(a) == canonical(b)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def compare(expected, actual, path="root", atol=1e-10, rtol=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), f"Keys differ: {path}")
        for key, value in expected.items():
            compare(value, actual[key], f"{path}.{key}", atol, rtol)
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(expected) == len(actual), f"Length differs: {path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{i}]", atol, rtol)
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol), f"Number differs: {path}: {expected} / {actual}")
    else:
        require(type(expected) is type(actual) and expected == actual, f"Value differs: {path}")


def transition(state, action, world):
    require(type(action) is int and 0 <= action < 4, "Invalid action")
    x, y, key = state
    dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[action]
    dest = (x + dx, y + dy)
    blocked = (not all(0 <= z <= 4 for z in dest)
               or dest in {tuple(p) for p in world["walls"]}
               or (dest == tuple(world["door"]) and key == 0))
    return tuple(state) if blocked else (*dest, int(key == 1 or dest == tuple(world["key"])))


def reference_maps():
    layouts = []
    for door in range(5):
        for kx in range(2):
            for ky in range(5):
                if (kx, ky) != (0, 0) and (door, kx, ky) != (2, 0, 4):
                    layouts.append((h(f"wm-active-v1:20261006:{door}:{kx}:{ky}"), door, kx, ky))
    require(len(layouts) == 44, "Layout population differs")
    result = []
    for i, (key_hash, door, kx, ky) in enumerate(sorted(layouts)[:12]):
        def rotated(x, y):
            for _ in range(i % 4):
                x, y = 4-y, x
            return [x, y]
        world = {"start": rotated(0, 0)+[0], "key": rotated(kx, ky),
                 "door": rotated(2, door), "walls": sorted(rotated(2, y) for y in range(5) if y != door)}
        closure = {tuple(world["start"])}
        while True:
            extended = closure | {transition(s, a, world) for s in closure for a in range(4)}
            if closure == extended:
                break
            closure = extended
        states = sorted(closure)
        require(len(states) == 30, "Unexpected reachable support")
        table = [[states.index(transition(s, a, world)) for a in range(4)] for s in states]
        map_id = f"map_{i:02d}"
        split = {name: [] for name in ("train", "query", "audit")}
        for a in range(4):
            ordered = sorted((4*s+a for s in range(30)), key=lambda row: h(f"wm-active-v1:split:{map_id}:{row}"))
            for name, indices in (("train", ordered[:18]), ("query", ordered[18:24]), ("audit", ordered[24:])):
                split[name].extend(indices)
        result.append({"id": map_id, "index": i, "phase": "engineering" if i < 4 else "evaluation",
                       "layout_hash": key_hash, "base_layout": {"door_y": door, "key_x": kx, "key_y": ky},
                       "clockwise_quarter_turns": i % 4, "world": world, "states": [list(s) for s in states],
                       "transitions": table, "goals": [list(p) for p in sorted({s[:2] for s in states})],
                       "splits": {name: sorted(ids) for name, ids in split.items()}})
    return result


def probabilities(logits):
    x = np.asarray(logits)
    require(x.shape == (30, 4, 30) and x.dtype.kind == "f" and np.isfinite(x).all(), "Invalid logit tensor")
    x = x.astype(np.float64)
    e = np.exp(x-np.max(x, axis=2, keepdims=True))
    return e/np.sum(e, axis=2, keepdims=True)


def validate_probability(P):
    P = np.asarray(P, dtype=np.float64)
    require(P.ndim == 3 and P.shape[1] == 4 and P.shape[0] == P.shape[2]
            and np.isfinite(P).all() and np.all(P >= 0) and np.all(P <= 1)
            and np.allclose(P.sum(axis=2), 1, rtol=0, atol=TOL), "Invalid probability tensor")
    return P


def value_and_policy(P, states, goal):
    n = len(states)
    goal_mask = np.array([list(s[:2]) == list(goal) for s in states])
    v = np.asarray([abs(s[0]-goal[0])+abs(s[1]-goal[1]) for s in states], dtype=np.float64)
    for _ in range(15):
        q = 1 + np.einsum("san,n->sa", P, v, optimize=False)
        v = np.min(q, axis=1)
        v[goal_mask] = 0
    q = 1 + np.einsum("san,n->sa", P, v, optimize=False)
    is_tied = q <= np.min(q, axis=1, keepdims=True)+TOL
    pi = np.asarray([np.flatnonzero(row)[0] for row in is_tied], dtype=np.int64)
    require(pi.shape == (n,), "Policy shape")
    return v, q, pi, is_tied.sum(axis=1), goal_mask


def local_information_value(distribution, action_values, continuation, action):
    alternative = min(v for j, v in enumerate(action_values) if j != action)
    return float(min(action_values) - np.dot(distribution, np.minimum(alternative, 1+continuation)))


def selector(P, states, goals, remaining, method, map_id, seed):
    P = validate_probability(P)
    remaining = sorted(remaining)
    require(remaining and len(set(remaining)) == len(remaining)
            and all(type(r) is int and 0 <= r < 4*len(states) for r in remaining), "Invalid remaining rows")
    require(seed in SEEDS and method in ARMS[:3], "Invalid selector identity")
    scores = {row: 0.0 for row in remaining}
    clamp_count, minimum = 0, 0.0
    if method == "random":
        ordering = sorted(remaining, key=lambda r: h(f"wm-active-v1:random:{map_id}:{seed}:{r}"))
        scores = {r: float(len(ordering)-ordering.index(r)) for r in remaining}
    elif method == "uncertainty":
        for row in remaining:
            p = P.reshape(-1, P.shape[-1])[row]
            scores[row] = float(-np.sum(p[p > 0]*np.log(p[p > 0])))
    else:
        n_pairs = sum(list(s[:2]) != list(g) for s in states for g in goals)
        require(n_pairs > 0, "No query-independent planning goals")
        for goal in goals:
            v, q, pi, _, goal_mask = value_and_policy(P, states, goal)
            mass = np.where(goal_mask, 0.0, 1/n_pairs)
            occupancy = np.zeros(len(states), dtype=np.float64)
            policy_matrix = P[np.arange(len(states)), pi]
            for _ in range(40):
                occupancy += mass
                mass = mass @ policy_matrix
                mass[goal_mask] = 0
            for row in remaining:
                s, a = divmod(row, 4)
                gain = local_information_value(P[s, a], q[s], v, a)
                minimum = min(minimum, gain)
                require(gain >= -TOL, "Negative decision value beyond roundoff")
                if gain < 0:
                    clamp_count += 1
                    gain = 0.0
                scores[row] += float(occupancy[s]*gain)
    maximum = max(scores.values())
    tied = sorted((r for r in remaining if scores[r] >= maximum-TOL),
                  key=lambda r: h(f"wm-active-v1:tie:{map_id}:{seed}:{r}"))
    return {"selected": tied[0], "scores": {str(r): scores[r] for r in remaining}, "tied": tied,
            "clamped_negative_gains": clamp_count, "minimum_raw_gain": minimum}


def distances(table, states, goal):
    # Multi-source reverse breadth first search; unreachable states stay None.
    reverse = [set() for _ in states]
    for s, row in enumerate(table):
        for t in row:
            reverse[t].add(s)
    result = [None]*len(states)
    queue = deque(i for i, s in enumerate(states) if list(s[:2]) == list(goal))
    for s in queue:
        result[s] = 0
    while queue:
        current = queue.popleft()
        for prior in reverse[current]:
            if result[prior] is None:
                result[prior] = result[current]+1
                queue.append(prior)
    return result


def navigation(P, m):
    P = validate_probability(P)
    states, table = m["states"], m["transitions"]
    rows = []
    for goal in m["goals"]:
        _, _, pi, ties, _ = value_and_policy(P, states, goal)
        ds = distances(table, states, goal)
        for start, state in enumerate(states):
            if list(state[:2]) == list(goal):
                continue
            require(ds[start] is not None, "Unreachable evaluation goal")
            path, actions, counts, visited = [start], [], [], {start}
            success = cycle = False
            for _ in range(40):
                s = path[-1]
                a = int(pi[s])
                actions.append(a)
                counts.append(int(ties[s]))
                t = int(table[s][a])
                path.append(t)
                if list(states[t][:2]) == list(goal):
                    success = True
                    break
                if t in visited:
                    cycle = True
                    break
                visited.add(t)
            rows.append({"optimal_steps": ds[start], "start": start, "goal": list(goal), "states": path,
                         "actions": actions, "tie_counts": counts, "success": success, "cycle": cycle, "steps": len(actions)})
    require(len(rows) == 600, "Evaluation population differs")
    good = [r for r in rows if r["success"]]
    summary = {"n": 600, "successes": len(good), "success_rate": len(good)/600,
               "cycles": sum(r["cycle"] for r in rows),
               "mean_steps_success": sum(r["steps"] for r in good)/len(good) if good else None,
               "mean_excess_steps_success": sum(r["steps"]-r["optimal_steps"] for r in good)/len(good) if good else None}
    return summary, rows


def prediction_metrics(P, m):
    indices = np.asarray(m["splits"]["audit"], dtype=np.int64)
    require(len(indices) == 24, "Final audit split count")
    selected = np.asarray(P).reshape(120, 30)[indices]
    target = np.asarray(m["transitions"], dtype=np.int64).reshape(120)[indices]
    true = selected[np.arange(24), target]
    maxima = selected == np.max(selected, axis=1, keepdims=True)
    correct = maxima[np.arange(24), target] & (maxima.sum(axis=1) == 1)
    return {"n": 24, "correct": int(correct.sum()),
            "capped_nll": float(np.mean(-np.log(np.maximum(true, np.finfo(np.float64).tiny)))),
            "zero_true_probability": int(np.count_nonzero(true == 0)), "mean_true_probability": float(np.mean(true))}


def expected_update_batches(train, acquired, seed, round_number, replay=False):
    require(round_number in (1, 2, 3, 4), "Invalid update round")
    generator = np.random.default_rng(seed+100000*round_number)
    old = generator.choice(train, size=(250, 16), replace=True)
    uniform = generator.random((250, 16))
    buffer = train if replay else acquired
    require(len(buffer) > 0, "Empty update observation buffer")
    new = np.asarray(buffer, dtype=np.int64)[(uniform*len(buffer)).astype(np.int64)]
    return np.hstack((old, new))


def validate_training(record, batches, target, updates, extra):
    expected_keys = {"updates", "loss_trace", "used_targets_sha256", "seconds"} | set(extra)
    require(set(record) == expected_keys, "Training record keys differ")
    require(type(record["updates"]) is int and record["updates"] == updates, "Update count differs")
    for key, value in extra.items():
        require(same(record[key], value), f"Training identity differs: {key}")
    require(type(record["seconds"]) in (float, int) and math.isfinite(record["seconds"])
            and 0 <= record["seconds"] <= 1800, "Invalid training duration")
    expected_targets = np.asarray(target, dtype=np.int64)[batches]
    require(record["used_targets_sha256"] == hashlib.sha256(expected_targets.tobytes()).hexdigest(), "Batch target digest differs")
    require(len(record["loss_trace"]) == updates//250, "Loss trace count")
    for step, row in enumerate(record["loss_trace"], 1):
        require(set(row) == {"update", "loss"} and type(row["update"]) is int and row["update"] == step*250
                and type(row["loss"]) in (float, int) and math.isfinite(row["loss"]) and row["loss"] >= 0, "Invalid declared loss trace")


def inventory(directory):
    directory = Path(directory)
    require(not any(p.is_symlink() for p in directory.rglob("*")), "Symlink in archive")
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def validate_manifest(directory, mapping, expected_names):
    require(isinstance(mapping, dict) and set(mapping) == set(expected_names), "Manifest coverage differs")
    for name, sha in mapping.items():
        p = Path(name)
        require(not p.is_absolute() and ".." not in p.parts and len(sha) == 64, "Invalid artifact reference")
        require(digest(Path(directory)/p) == sha, f"Artifact hash differs: {name}")


def load_array(path, dtype, shape):
    value = np.load(path, allow_pickle=False)
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and value.shape == shape,
            f"Array descriptor differs: {path}")
    require(np.isfinite(value).all(), f"Nonfinite saved array: {path}")
    return value


def validate_plan(directory):
    directory = Path(directory)
    plan = read(directory/"plan.json")
    require(plan["schema"] == "world-model-acquisition-v1", "Plan schema")
    require(same(plan["maps"], reference_maps()), "World, split or map order differs")
    require(same(plan["seeds"], list(SEEDS)) and same(plan["arms"], list(ARMS))
            and same(plan["budgets"], [0, 1, 2, 4]) and type(plan["rounds"]) is int and plan["rounds"] == 4, "Fixed panel differs")
    require(same(plan["training"], {"initial_steps": 1500, "extra_steps_per_round": 250, "batch_size": 32,
            "original_slots": 16, "acquired_slots": 16, "learning_rate": .003, "optimizer": "Adam",
            "weight_decay": 0, "device": "cpu", "threads": 2, "dtype": "float32"}), "Training contract differs")
    require(same(plan["planner"], {"horizon": 16, "episode_cap": 40, "tie_tolerance": TOL, "full_state_cycle_stop": True}), "Planner contract differs")
    require(same(plan["resources"], {"timeout_seconds_per_phase": 1800, "rss_bytes": 2*2**30,
            "output_bytes": 2**30, "min_free_bytes": 8*2**30}), "Resource contract differs")
    for key, value in (("split_hash", "wm-active-v1:split:{map_id}:{row_id}"),
                       ("tie_hash", "wm-active-v1:tie:{map_id}:{seed}:{row_id}"),
                       ("random_hash", "wm-active-v1:random:{map_id}:{seed}:{row_id}")):
        require(plan[key] == value, f"Hash namespace differs: {key}")
    require(set(plan["source_sha256"]) == {"world.py", "selection.py", "models.py", "run.py", "launch.py", "audit.py"}, "Source coverage")
    here = Path(__file__).resolve().parent
    for name, sha in plan["source_sha256"].items():
        require(digest(here/name) == digest(directory/"sources"/name) == sha, f"Source differs: {name}")
    require(digest(here/"README.md") == digest(directory/"sources"/"README.md") == plan["protocol_sha256"], "Protocol source differs")
    require(plan["runtime"] == {"python": sys.version.split()[0], "numpy": np.__version__,
            "torch": importlib.metadata.version("torch")}, "Numerical runtime differs")
    require(plan["budget"]["new_provider_calls"] == 0 and plan["budget"]["new_estimated_usd"] == 0, "Unexpected provider work")
    return plan, digest(directory/"plan.json")


def expected_phase_names(maps):
    acquisition = {"FREEZE.json", "INITIAL_TRAINING_COMPLETE.json"}
    for m in maps:
        for seed in SEEDS:
            for kind in KINDS:
                prefix = f"{m['id']}/{kind}_{seed}"
                acquisition.update(f"{prefix}/{name}" for name in ("base.pt", "base.npy", "base_batches.npy", "base_training.json"))
                for arm in ARMS:
                    for rnd in range(1, 5):
                        acquisition.update(f"{prefix}/{arm}/round_{rnd}{suffix}" for suffix in
                                           (".pt", ".npy", "_batches.npy", "_query.json", "_training.json"))
    collected = acquisition | {"ACQUISITION_COMPLETE.json", "report.json"}
    for m in maps:
        collected |= {f"{m['id']}/episodes.jsonl.gz", f"{m['id']}/report.json"}
    return acquisition, collected


def audit_acquisition(phase_dir, maps):
    initial = read(phase_dir/"INITIAL_TRAINING_COMPLETE.json")
    complete = read(phase_dir/"ACQUISITION_COMPLETE.json")
    require(set(initial) == {"at_utc", "fits"} and set(complete) == {"at_utc", "chains", "files_sha256"}, "Acquisition receipts")
    records, chains, predictions = [], [], {}
    for m in maps:
        targets = np.asarray(m["transitions"], dtype=np.int64).reshape(120)
        train, query = m["splits"]["train"], m["splits"]["query"]
        for seed in SEEDS:
            for kind in KINDS:
                name = f"{kind}_{seed}"
                folder = phase_dir/m["id"]/name
                base_batches = load_array(folder/"base_batches.npy", "int64", (1500, 32))
                wanted = np.random.default_rng(seed).choice(train, (1500, 32), replace=True)
                require(np.array_equal(base_batches, wanted), "Initial sample stream differs")
                record = read(folder/"base_training.json")
                validate_training(record, base_batches, targets, 1500, {"map": m["id"], "model": name})
                records.append(record)
                require((folder/"base.pt").stat().st_size > 0, "Empty initial checkpoint")
                base = load_array(folder/"base.npy", "float32", (30, 4, 30))
                predictions[(m["id"], name, "base", 0)] = probabilities(base)
                for arm in ARMS:
                    acquired, previous = [], folder/"base.npy"
                    P = predictions[(m["id"], name, "base", 0)]
                    for rnd in range(1, 5):
                        stem = folder/arm/f"round_{rnd}"
                        decision = read(str(stem)+"_query.json")
                        remaining = sorted(set(query)-set(acquired))
                        if arm == "replay":
                            expected = {"selected": None, "revealed_target": None, "scores": {}, "tied": [],
                                        "clamped_negative_gains": 0, "minimum_raw_gain": 0.0}
                        else:
                            expected = selector(P, m["states"], m["goals"], remaining, arm, m["id"], seed)
                            chosen = expected["selected"]
                            require(chosen in query and chosen not in acquired, "Illegal/repeated observation")
                            acquired.append(chosen)
                            expected["revealed_target"] = int(targets[chosen])
                        expected.update(round=rnd, method=arm, seed=seed, map=m["id"], remaining_before=remaining,
                                        acquired=list(acquired), prediction_sha256=digest(previous), seconds=decision.get("seconds"))
                        require(type(expected["seconds"]) in (float, int) and math.isfinite(expected["seconds"])
                                and 0 <= expected["seconds"] <= 1800, "Selector duration")
                        compare(expected, decision, f"{m['id']}/{name}/{arm}/{rnd}/decision")
                        batches = load_array(str(stem)+"_batches.npy", "int64", (250, 32))
                        require(np.array_equal(batches, expected_update_batches(train, acquired, seed, rnd, arm == "replay")), "Update sample stream differs")
                        require(set(batches.reshape(-1)).issubset(set(train)|set(acquired)), "Unobserved targets used by batch")
                        validate_training(read(str(stem)+"_training.json"), batches, targets, 250, {"round": rnd, "queries": len(acquired)})
                        require(Path(str(stem)+".pt").stat().st_size > 0, "Empty updated checkpoint")
                        previous = Path(str(stem)+".npy")
                        P = probabilities(load_array(previous, "float32", (30, 4, 30)))
                        predictions[(m["id"], name, arm, rnd)] = P
                    chains.append({"map": m["id"], "model": name, "arm": arm, "queries": acquired})
    require(same(records, initial["fits"]), "Initial-fit receipt differs")
    require(same(chains, complete["chains"]), "Acquisition chain receipt differs")
    return predictions, chains


def audit_metrics(phase_dir, maps, predictions, chains):
    reports = []
    episode_count = 0
    for m in maps:
        report = {"map": m["id"], "phase": m["phase"], "models": {}}
        with gzip.open(phase_dir/m["id"]/"episodes.jsonl.gz", "rt") as stream:
            def check(P, name, arm, budget):
                nonlocal episode_count
                summary, rows = navigation(P, m)
                for row in rows:
                    line = stream.readline()
                    require(bool(line), "Missing episode")
                    require(same({"model": name, "arm": arm, "budget": budget, **row}, json.loads(line)), "Episode identity/policy/path differs")
                    episode_count += 1
                return {"planning": summary, "prediction_audit": prediction_metrics(P, m)}
            report["oracle"] = check(np.eye(30, dtype=np.float64)[np.asarray(m["transitions"])], "oracle", "oracle", 0)
            require(report["oracle"]["planning"]["successes"] == 600 and report["oracle"]["planning"]["mean_excess_steps_success"] == 0, "Oracle graph control failed")
            for seed in SEEDS:
                for kind in KINDS:
                    name = f"{kind}_{seed}"
                    result = {"base": check(predictions[(m["id"], name, "base", 0)], name, "base", 0), "arms": {}}
                    for arm in ARMS:
                        result["arms"][arm] = {str(b): check(predictions[(m["id"], name, arm, b)], name, arm, b) for b in (1, 2, 4)}
                    report["models"][name] = result
            require(not stream.read(1), "Extra episode rows")
        differences = [r["arms"]["decision"]["4"]["planning"]["success_rate"]-r["arms"]["random"]["4"]["planning"]["success_rate"] for r in report["models"].values()]
        report["primary_difference"] = float(np.mean(differences))
        report["query_overlap"] = []
        for seed in SEEDS:
            for kind in KINDS:
                name = f"{kind}_{seed}"
                chosen = {c["arm"]: set(c["queries"]) for c in chains if c["map"] == m["id"] and c["model"] == name}
                report["query_overlap"].append({"model": name,
                    "decision_random": len(chosen["decision"] & chosen["random"]),
                    "decision_uncertainty": len(chosen["decision"] & chosen["uncertainty"]),
                    "uncertainty_random": len(chosen["uncertainty"] & chosen["random"])})
        compare(report, read(phase_dir/m["id"]/"report.json"), m["id"])
        reports.append(report)
    total = read(phase_dir/"report.json")
    expected = {"status": "COLLECTED_PENDING_AUDIT", "phase": maps[0]["phase"], "maps": reports,
                "primary_mean_difference": float(np.mean([r["primary_difference"] for r in reports])),
                "positive_maps": sum(r["primary_difference"] > 0 for r in reports),
                "seconds": total.get("seconds"), "peak_rss_bytes": total.get("peak_rss_bytes"),
                "new_provider_calls": 0, "new_estimated_usd": 0}
    require(type(expected["seconds"]) in (float, int) and math.isfinite(expected["seconds"])
            and 0 <= expected["seconds"] <= 1800, "Collector elapsed bound")
    require(type(expected["peak_rss_bytes"]) is int and 0 < expected["peak_rss_bytes"] <= 2*2**30, "Collector RSS bound")
    compare(expected, total, "phase report")
    return reports, episode_count


def audit(directory, phase):
    directory = Path(directory).resolve()
    require(phase in ("engineering", "evaluation"), "Unknown phase")
    phase_dir = directory/phase
    require(not (phase_dir/"STOP.json").exists(), "Original STOP remains STOP; no grading")
    require(not (phase_dir/"AUDIT.json").exists(), "Audit already exists; no overwrite/retry")
    plan, plan_sha = validate_plan(directory)
    watchdog_path = directory/f"{phase}_WATCHDOG.json"
    watchdog = read(watchdog_path)
    require(watchdog["status"] == "PASS_COLLECTOR_EXIT" and watchdog["phase"] == phase
            and watchdog["returncode"] == 0 and watchdog["reason"] is None and watchdog["retries"] == 0
            and 0 <= watchdog["seconds"] <= 1800 and 0 < watchdog["peak_sampled_rss_bytes"] <= 2*2**30,
            "External owner did not close successfully")
    maps = [m for m in plan["maps"] if m["phase"] == phase]
    require(len(maps) == (4 if phase == "engineering" else 8), "Phase family count")
    before = inventory(phase_dir)
    acquisition_names, collected_names = expected_phase_names(maps)
    require(set(before) == collected_names | {"COLLECTED.json"}, "Incomplete or unexpected phase archive")
    collected = read(phase_dir/"COLLECTED.json")
    require(set(collected) == {"at_utc", "status", "files_sha256"} and collected["status"] == "COLLECTED_PENDING_AUDIT", "Collection terminal status")
    validate_manifest(phase_dir, collected["files_sha256"], collected_names)
    acquisition = read(phase_dir/"ACQUISITION_COMPLETE.json")
    validate_manifest(phase_dir, acquisition["files_sha256"], acquisition_names)
    freeze = read(phase_dir/"FREEZE.json")
    require(freeze["phase"] == phase and freeze["plan_sha256"] == plan_sha
            and freeze["source_sha256"] == plan["source_sha256"], "Phase freeze differs")
    if phase == "evaluation":
        old = read(directory/"engineering"/"AUDIT.json")
        require(old["status"] == "PASS" and old["phase"] == "engineering" and old["plan_sha256"] == plan_sha
                and old["collected_sha256"] == digest(directory/"engineering"/"COLLECTED.json")
                and freeze["engineering_audit_sha256"] == digest(directory/"engineering"/"AUDIT.json"), "Engineering admission differs")
    dates = [freeze["at_utc"], read(phase_dir/"INITIAL_TRAINING_COMPLETE.json")["at_utc"], acquisition["at_utc"], collected["at_utc"]]
    times = [datetime.fromisoformat(v) for v in dates]
    require(all(t.tzinfo is not None for t in times) and times == sorted(times), "Phase chronology differs")
    # All archive/collection identity checks precede any evaluation of quality.
    predictions, chains = audit_acquisition(phase_dir, maps)
    reports, episode_count = audit_metrics(phase_dir, maps, predictions, chains)
    require(inventory(phase_dir) == before and digest(directory/"plan.json") == plan_sha, "Archive changed during audit")
    validate_plan(directory)
    result = {"status": "PASS", "schema": "world-model-acquisition-independent-audit-v1", "phase": phase,
              "at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_sha,
              "collected_sha256": digest(phase_dir/"COLLECTED.json"), "report_sha256": digest(phase_dir/"report.json"),
              "watchdog_sha256": digest(watchdog_path), "audit_code_sha256": digest(__file__),
              "counts": {"maps": len(maps), "initial_fits": 6*len(maps), "update_chains": 24*len(maps),
                         "update_segments": 96*len(maps), "initial_optimizer_steps": 9000*len(maps),
                         "additional_optimizer_steps": 24000*len(maps), "queried_transitions": 72*len(maps),
                         "saved_prediction_arrays": 102*len(maps), "episodes": episode_count},
              "primary_map_differences": {r["map"]: r["primary_difference"] for r in reports},
              "primary_mean_difference": float(np.mean([r["primary_difference"] for r in reports])),
              "limits": ["Independent NumPy/world/selector/planner recount; no production imports or model execution.",
                         "Checkpoint, batch, declared-target and source provenance verified; gradients and optimizer tensors not replayed.",
                         "Fixed eight-map evaluation is descriptive; seeds and 600 tasks are not independent maps.",
                         "DI is frozen-value row-revelation sensitivity, not expected SGD benefit; confident errors may score zero."]}
    with (phase_dir/"AUDIT.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("engineering", "evaluation"), required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.out, args.phase), allow_nan=False))
