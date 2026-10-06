"""Independent finite-world arithmetic and saved-result verifier.

No training/model imports. This does not replay optimization or claim independent
world generalization.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

import numpy as np

AUDIT_SCHEMA = "world-model-planning-independent-audit-v1"
SEEDS = (11, 29, 47)
MODEL_NAMES = tuple(f"{kind}_{seed}" for seed in SEEDS for kind in ("direct", "energy"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def same(left, right):
    return canonical(left) == canonical(right)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition(state, action):
    """Literal fixed rules, implemented independently of world.py."""
    require(type(action) is int and 0 <= action < 4, "Invalid action")
    x, y, key = state
    destinations = ((x, y-1), (x+1, y), (x, y+1), (x-1, y))
    xx, yy = destinations[action]
    outside = xx < 0 or xx > 4 or yy < 0 or yy > 4
    wall = xx == 2 and yy != 2
    locked = (xx, yy) == (2, 2) and key == 0
    if outside or wall or locked:
        return tuple(state)
    return (xx, yy, 1 if key == 1 or (xx, yy) == (0, 4) else 0)


def reference_world():
    # Closure by repeated full-set expansion, not the primary BFS implementation.
    reached = {(0, 0, 0)}
    while True:
        expanded = reached | {transition(s, a) for s in reached for a in range(4)}
        if expanded == reached:
            break
        reached = expanded
    states = sorted(reached)
    require(len(states) == 30, "Unexpected reachable closure")
    indices = {s: i for i, s in enumerate(states)}
    table = np.asarray([[indices[transition(s, a)] for a in range(4)] for s in states], dtype=np.int64)
    return states, table


def reference_split(states):
    indices = {tuple(s): i for i, s in enumerate(states)}
    test_forced = {4*indices[(0, 3, 0)]+2, 4*indices[(1, 2, 1)]+1}
    train_forced = {4*indices[(1, 4, 0)]+3, 4*indices[(3, 2, 1)]+3}
    prng = random.Random(20261006)
    test = []
    for a in range(4):
        available = [i for i in range(a, 120, 4) if i not in test_forced and i not in train_forced]
        prng.shuffle(available)
        mandatory = sorted(test_forced & set(range(a, 120, 4)))
        test.extend(mandatory + available[:6-len(mandatory)])
    return [i for i in range(120) if i not in test], sorted(test)


def goal_distances(table, states, goal):
    """Reverse multi-source BFS; both key states at a goal are terminals."""
    reverse = [[] for _ in states]
    for s, row in enumerate(table):
        for nxt in set(int(v) for v in row):
            reverse[nxt].append(s)
    distances = [None] * len(states)
    queue = deque()
    for s, state in enumerate(states):
        if tuple(state[:2]) == tuple(goal):
            distances[s] = 0
            queue.append(s)
    while queue:
        nxt = queue.popleft()
        for s in reverse[nxt]:
            if distances[s] is None:
                distances[s] = distances[nxt] + 1
                queue.append(s)
    return distances


def validate_plan(plan):
    require(plan["schema"] == "world-model-planning-pilot-v1", "Wrong schema")
    states, table = reference_world()
    require(same(plan["states"], states), "State vocabulary differs")
    require(same(plan["transitions"], table.tolist()), "Transition oracle differs")
    required_world = {"width": 5, "height": 5, "start": [0, 0, 0], "key": [0, 4],
                      "door": [2, 2], "walls": [[2, 0], [2, 1], [2, 3], [2, 4]],
                      "actions": ["north", "east", "south", "west"],
                      "pickup": "automatic_permanent", "door_rule": "requires_key"}
    require(same(plan["world"], required_world), "World declaration differs")
    train, test = reference_split(states)
    require(same(plan["train"], train) and same(plan["test"], test), "Split differs")
    require(plan["split_seed"] == 20261006 and plan["rollout_seed"] == 20261007, "Seed differs")
    require(same(plan["horizons"], [1, 3, 16]) and same(plan["seeds"], list(SEEDS)), "Fixed panel differs")
    require(plan["episode_cap"] == 40 and plan["tie_tolerance"] == 1e-12
            and plan["stop_on_first_true_state_cycle"] is True, "Planner contract differs")
    require(same(plan["rollout_prefixes"], [1, 3, 6, 10]), "Rollout horizons differ")
    goals = sorted({s[:2] for s in states})
    distances = {g: goal_distances(table, states, g) for g in goals}
    cases = []
    for start, state in enumerate(states):
        for goal in goals:
            if state[:2] != goal:
                distance = distances[goal][start]
                require(distance is not None, "Unreachable task")
                cases.append({"start": start, "goal": list(goal), "optimal_steps": distance})
    require(len(cases) == 600 and same(plan["planning_cases"], cases), "Episode census differs")
    prng = random.Random(20261007)
    rollouts = [{"start": s, "replicate": r, "actions": [prng.randrange(4) for _ in range(10)]}
                for s in range(30) for r in range(4)]
    require(same(plan["rollout_cases"], rollouts), "Open-loop census differs")
    return states, table, cases, rollouts


def probability_from_logits(logits):
    require(isinstance(logits, np.ndarray) and logits.dtype == np.dtype("float32")
            and logits.shape == (30, 4, 30), "Prediction shape/dtype differs")
    require(np.isfinite(logits).all(), "Nonfinite logits")
    x = logits.astype(np.float64)
    probabilities = np.empty_like(x)
    for s in range(30):
        for a in range(4):
            row = x[s, a]
            numerator = [math.exp(float(v - max(row))) for v in row]
            probabilities[s, a] = np.asarray(numerator) / math.fsum(numerator)
    return probabilities


def bellman_policy(probabilities, states, goal, horizon):
    """Independent row-wise accumulation; no primary planner import."""
    count = len(states)
    require(probabilities.shape == (count, 4, count), "Planner matrix shape differs")
    values = np.asarray([abs(s[0]-goal[0]) + abs(s[1]-goal[1]) for s in states], dtype=np.float64)
    terminals = [i for i, s in enumerate(states) if tuple(s[:2]) == tuple(goal)]
    for _ in range(horizon):
        q = np.asarray([[1 + math.fsum(float(p)*float(v) for p, v in zip(probabilities[s, a], values))
                         for a in range(4)] for s in range(count)], dtype=np.float64)
        values = q.min(axis=1)
        values[terminals] = 0
    ties = q <= q.min(axis=1, keepdims=True) + 1e-12
    actions = [next(a for a in range(4) if ties[s, a]) for s in range(count)]
    return actions, ties.sum(axis=1).tolist(), q


def replay_episode(table, states, start, goal, actions_by_state, ties_by_state):
    trace, actions, ties = [start], [], []
    seen = {start}
    current = start
    success = tuple(states[current][:2]) == tuple(goal)
    cycle = False
    while not success and not cycle and len(actions) < 40:
        action = int(actions_by_state[current])
        actions.append(action)
        ties.append(int(ties_by_state[current]))
        current = int(table[current, action])
        trace.append(current)
        success = tuple(states[current][:2]) == tuple(goal)
        cycle = not success and current in seen
        seen.add(current)
    return {"states": trace, "actions": actions, "tie_counts": ties,
            "success": success, "cycle": cycle, "steps": len(actions)}


def categorical_metrics(probabilities, targets):
    require(probabilities.ndim == 2 and len(probabilities) == len(targets), "Metric shape differs")
    nll, true_probabilities, correct, ties, zeros = [], [], 0, 0, 0
    for row, target in zip(probabilities, targets):
        require(np.isfinite(row).all() and min(row) >= 0 and abs(math.fsum(row)-1) <= 1e-10,
                "Invalid distribution")
        winners = np.flatnonzero(row == row.max()).tolist()
        ties += len(winners) != 1
        correct += len(winners) == 1 and winners[0] == int(target)
        probability = float(row[int(target)])
        zeros += probability == 0
        true_probabilities.append(probability)
        nll.append(-math.log(max(probability, np.finfo(np.float64).tiny)))
    return {"n": len(targets), "capped_nll": math.fsum(nll)/len(nll),
            "zero_true_probability": int(zeros), "true_probability": math.fsum(true_probabilities)/len(targets),
            "correct": int(correct), "ties": int(ties), "accuracy": correct/len(targets)}


def rollout_metrics(probabilities, table, cases, prefixes=(1, 3, 6, 10)):
    by_prefix = {prefix: ([], []) for prefix in prefixes}
    for case in cases:
        distribution = np.zeros(len(table), dtype=np.float64)
        distribution[case["start"]] = 1
        actual = case["start"]
        for depth, action in enumerate(case["actions"], 1):
            distribution = np.asarray([math.fsum(float(distribution[s])*float(probabilities[s, action, nxt])
                                                for s in range(len(table))) for nxt in range(len(table))])
            actual = int(table[actual, action])
            if depth in by_prefix:
                by_prefix[depth][0].append(distribution.copy())
                by_prefix[depth][1].append(actual)
    return {str(depth): categorical_metrics(np.asarray(rows), targets)
            for depth, (rows, targets) in by_prefix.items()}


def compare_numeric(actual, expected, path="root", atol=1e-10, rtol=1e-9):
    """Strict containers/counts; fixed floating tolerance, no missing-field waiver."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Keys differ: {path}")
        for key in expected:
            compare_numeric(actual[key], expected[key], f"{path}.{key}", atol, rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Length differs: {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_numeric(a, e, f"{path}[{i}]", atol, rtol)
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and abs(actual-expected) <= atol+rtol*abs(expected), f"Float differs: {path}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Value differs: {path}")


def planning_summary(rows):
    successes = [row for row in rows if row["success"]]
    bins = {"1-3": [], "4-6": [], "7+": []}
    for row in rows:
        distance = row["optimal_steps"]
        bins["1-3" if distance <= 3 else "4-6" if distance <= 6 else "7+"].append(row)
    return {"n": len(rows), "successes": len(successes), "success_rate": len(successes)/len(rows),
            "cycles": sum(row["cycle"] for row in rows),
            "mean_steps_success": math.fsum(row["steps"] for row in successes)/len(successes) if successes else None,
            "mean_excess_steps_success": math.fsum(row["steps"]-row["optimal_steps"] for row in successes)/len(successes) if successes else None,
            "tied_decisions": sum(sum(value > 1 for value in row["tie_counts"]) for row in rows),
            "decisions": sum(row["steps"] for row in rows),
            "by_distance": {label: {"n": len(group), "successes": sum(r["success"] for r in group)}
                            for label, group in bins.items()}}


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def validate_fit_metadata(plan, fits, training, out):
    require([r["model"] for r in fits] == list(MODEL_NAMES), "Six fit order differs")
    require(training["all_six_complete"] is True and same(training["fits"], fits), "Training phase differs")
    require(training["plan_sha256"] == digest(out/"plan.json"), "Training plan binding differs")
    expected = dict(steps=1500, batch_size=32, learning_rate=0.003, optimizer="Adam", weight_decay=0,
                    activation="GELU", dtype="float32", device="cpu", threads=2, direct_width=64,
                    energy_width=71, shared_batch_order_per_seed=True, all_six_fits_before_evaluation=True)
    require(same(plan["training"], expected), "Training contract differs")
    for record, name in zip(fits, MODEL_NAMES):
        kind, seed = name.split("_")
        require(record["seed"] == int(seed) and record["kind"] == kind and record["steps"] == 1500,
                "Fit identity/step count differs")
        # Literal layer sizes: direct 16->64->64->30; joint 28->71->71->1.
        count = (16+1)*64+(64+1)*64+(64+1)*30 if kind == "direct" else (28+1)*71+(71+1)*71+72
        require(record["parameters"] == count, "Parameter accounting differs")
        require(record["checkpoint_sha256"] == digest(out/f"{name}.pt"), "Checkpoint hash differs")
        require([r["update"] for r in record["training_loss_trace"]] == [250, 500, 750, 1000, 1250, 1500],
                "Training progress coverage differs")
        require(all(type(r["batch_loss"]) in (float, int) and math.isfinite(r["batch_loss"])
                    and r["batch_loss"] >= 0 for r in record["training_loss_trace"]), "Invalid loss metadata")
        require(math.isfinite(record["seconds"]) and record["seconds"] >= 0, "Invalid timing")


def audit(directory, launch_receipt):
    out = Path(directory).resolve()
    require(out.is_dir() and not (out/"STOP.json").exists(), "Missing collection or preserved STOP")
    require(not (out/"INDEPENDENT_AUDIT.json").exists(), "Audit already exists; no repeat invocation")
    required = {"plan.json", "OWNER.json", "FIT_PROGRESS.json", "TRAINING_COMPLETE.json", "predictions.npz",
                "episodes.json", "report.json", "COLLECTED.json"} | {f"{name}.pt" for name in MODEL_NAMES} | {
                    f"predictions-{name}.npy" for name in MODEL_NAMES}
    require({p.name for p in out.iterdir()} == required and all(p.is_file() for p in out.iterdir()),
            "Collection inventory differs")
    snapshot = {name: digest(out/name) for name in sorted(required)}
    collected, owner = read(out/"COLLECTED.json"), read(out/"OWNER.json")
    require(collected["status"] == "COLLECTED_PENDING_INDEPENDENT_AUDIT", "Collection not complete")
    require(owner["status"] == "EXITED_COLLECTION_COMPLETE", "Owner not closed")
    require(type(owner["pid"]) is int and owner["pid"] > 0, "Invalid owner PID")
    launch_path = Path(launch_receipt).resolve()
    require(out not in launch_path.parents, "Watchdog receipt must be outside sealed collector directory")
    launch_hash, launch = digest(launch_path), read(launch_path)
    require(launch["status"] == "PASS_COLLECTOR_EXIT" and launch["pid"] == owner["pid"]
            and launch["returncode"] == 0 and launch["reason"] is None and launch["retries"] == 0,
            "Watchdog did not close successfully")
    require(launch["timeout_seconds"] == 900 and launch["poll_seconds"] == 0.5
            and 0 <= launch["elapsed_seconds"] <= 900
            and 0 < launch["peak_sampled_rss_bytes"] <= 2*2**30, "Watchdog resource contract differs")
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Original collection PID is still active or inaccessible")
    expected_bound = required - {"OWNER.json", "COLLECTED.json"}
    require(collected["files_sha256"] == {name: snapshot[name] for name in expected_bound}, "Collection seals differ")
    require(collected["episodes"] == 12600 and collected["model_fits"] == 6
            and collected["prediction_rows"] == 720, "Collection count differs")
    require(0 <= collected["seconds"] <= 900 and 0 < collected["max_rss_bytes"] <= 2*2**30,
            "Reported resource cap exceeded")
    require(sum((out/name).stat().st_size for name in required) <= 128*2**20, "Archive cap exceeded")
    plan = read(out/"plan.json")
    require(same(plan["resources"], dict(timeout_seconds=900, max_rss_bytes=2*2**30,
                                       max_output_bytes=128*2**20, min_free_bytes=8*2**30)), "Resource contract differs")
    require(plan["runtime"]["python"] == sys.version.split()[0] and plan["runtime"]["numpy"] == np.__version__,
            "Audit Python/NumPy runtime differs")
    source_dir = Path(__file__).resolve().parent
    source_map = plan["source_sha256"]
    require(set(source_map) == {p.name for p in source_dir.glob("*.py")}, "Source inventory differs")
    require(all(digest(source_dir/name) == value for name, value in source_map.items()), "Source hash differs")
    require(collected["source_sha256"] == source_map, "Collection source binding differs")
    require(plan["prediction_format"] == "float32 categorical logits, shape [30 states,4 actions,30 candidates]. energy_* keys store NEGATIVE energies, not raw energies. Probability reconstruction uses float64 softmax.", "Prediction sign/shape contract differs")
    require(plan["nll_definition"] == "capped_nll = mean(-log(max(true_probability,float64 tiny))). Exact zero counts are reported separately; true NLL is infinite if any true probability is zero. No probability smoothing or renormalization is added.", "NLL contract differs")
    states, table, cases, rollout_cases = validate_plan(plan)
    fits, training = read(out/"FIT_PROGRESS.json"), read(out/"TRAINING_COMPLETE.json")
    validate_fit_metadata(plan, fits, training, out)
    require(0 <= training["elapsed_seconds"] <= collected["seconds"], "Training chronology differs")
    report, saved_episodes = read(out/"report.json"), read(out/"episodes.json")
    require(report["status"] == "COLLECTED_PENDING_INDEPENDENT_AUDIT" and same(report["fits"], fits), "Report status/fits differ")
    require(report["model_training_runs"] == 6 and report["provider_calls"] == 0
            and report["new_spending_usd"] == 0 and report["novelty_established"] is False, "Report scope differs")
    require(len(saved_episodes) == 12600 and set(report["models"]) == set(MODEL_NAMES) | {"oracle"}, "Result coverage differs")
    # Only now open the six complete raw prediction matrices. No checkpoint decoding.
    with np.load(out/"predictions.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(MODEL_NAMES), "Prediction keys differ")
        probability = {}
        for name in MODEL_NAMES:
            raw = archive[name]
            retained = np.load(out/f"predictions-{name}.npy", allow_pickle=False)
            require(retained.dtype == raw.dtype and retained.shape == raw.shape
                    and retained.tobytes() == raw.tobytes(), "Per-return/aggregate predictions differ")
            probability[name] = probability_from_logits(raw)
    probability["oracle"] = np.asarray([[[float(next_state == table[s, a]) for next_state in range(30)]
                                          for a in range(4)] for s in range(30)])
    expected_episodes, model_results = [], {}
    for name in (*MODEL_NAMES, "oracle"):
        p = probability[name]
        result = {"one_step": {}, "rollout": rollout_metrics(p, table, rollout_cases), "planning": {}}
        controls = {"native": p, "wrong_action": p[:, [1, 2, 3, 0], :],
                    "action_average": np.repeat(p.mean(axis=1, keepdims=True), 4, axis=1)}
        for control, matrix in controls.items():
            result["one_step"][control] = {}
            for split in ("train", "test"):
                indices = plan[split]
                result["one_step"][control][split] = categorical_metrics(matrix.reshape(120, 30)[indices],
                                                                         table.reshape(120)[indices])
        for horizon in (1, 3, 16):
            policies = {goal: bellman_policy(p, states, goal, horizon) for goal in sorted({tuple(c["goal"]) for c in cases})}
            episodes = []
            for case in cases:
                actions, ties, _ = policies[tuple(case["goal"])]
                episode = {"model": name, "horizon": horizon, **case,
                           **replay_episode(table, states, case["start"], case["goal"], actions, ties)}
                episodes.append(episode)
            result["planning"][str(horizon)] = planning_summary(episodes)
            expected_episodes.extend(episodes)
        model_results[name] = result
    require(same(saved_episodes, expected_episodes), "Saved episode action/state/tie/cycle/BFS records differ")
    compare_numeric(report["models"], model_results, "report.models")
    require(model_results["oracle"]["planning"]["16"]["successes"] == 600, "Exact-dynamics H16 positive control failed")
    require(snapshot == {name: digest(out/name) for name in snapshot}, "Collection changed during review")
    require(digest(launch_path) == launch_hash, "Watchdog receipt changed during review")
    require(all(digest(source_dir/name) == value for name, value in source_map.items()), "Source changed during review")
    return {"schema": AUDIT_SCHEMA, "status": "PASS", "implementation_pass": True,
            "at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": snapshot["plan.json"],
            "report_sha256": snapshot["report.json"], "collected_sha256": snapshot["COLLECTED.json"],
            "audit_code_sha256": digest(__file__), "source_sha256": source_map, "artifact_sha256": snapshot,
            "launch_receipt_sha256": launch_hash,
            "checks": {"reachable_states": 30, "unique_transitions": 120, "training_transitions": 96,
                       "heldout_transitions": 24, "models": 6, "reported_fits": 6, "reported_updates_per_fit": 1500,
                       "prediction_rows": 720, "episodes": 12600, "rollout_sequences_per_condition": 120,
                       "rollout_prefixes": [1, 3, 6, 10], "oracle_H16_successes": 600,
                       "owner_pid_inactive": True, "source_and_archive_unchanged": True},
            "numerical_comparison": {"absolute_tolerance": 1e-10, "relative_tolerance": 1e-9,
                                     "counts_categories_episodes": "exact", "action_tie_tolerance": 1e-12},
            "independence": "Separate rule/closure/reverse-BFS/Bellman/metric code; no primary helper or neural forward import. Final checkpoints and training progress are hash-bound and reconciled, not gradient-replayed. Independent scoring, not external experimental replication.",
            "resource_scope": "Bound external watchdog reports 0.5-second target polling of the collector PID and elapsed time; collector reports self-process high-water RSS. This is not continuous process-tree monitoring or an independently sampled minimum-free-space history.",
            "models": model_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = audit(args.directory, args.launch_receipt)
        with (args.directory/"INDEPENDENT_AUDIT.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "PASS", "path": str(args.directory/"INDEPENDENT_AUDIT.json"),
                          "sha256": digest(args.directory/"INDEPENDENT_AUDIT.json")}))
    except BaseException as exc:
        failure = args.directory/"AUDIT_FAILURE.json"
        if not failure.exists():
            with failure.open("x") as stream:
                json.dump({"status": "FAIL", "type": type(exc).__name__, "reason": str(exc),
                           "audit_code_sha256": digest(__file__)}, stream, indent=2, allow_nan=False)
                stream.write("\n")
        raise
