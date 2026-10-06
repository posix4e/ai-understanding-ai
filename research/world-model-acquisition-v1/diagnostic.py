"""Post-hoc direct_11 diagnostic from sealed predictions; no model or fit imports."""
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import argparse
import gzip
import hashlib
import importlib.util
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/world-model-planning/ACQUISITION_DIAGNOSTIC_V1'
WORK = ROOT / 'work/world-model-acquisition-v1'
ORIGINAL = ROOT / 'outputs/world-model-planning/V1'
PILOT_PLAN_SHA = '858492ca878ca59e4f36c90171f6e56bf93eab0dcf501ab417da43a577f8815a'
WORLD_SHA = 'f7e8a885794fd42bfad6760d2e09b47c1ea1e8a4d6e95b5c3028cf55e7580f3f'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def inventory(directory):
    return {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def verify_inventory(items):
    for rel, item in items.items():
        p = ROOT / rel
        require(p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'],
                'Changed binding: ' + rel)


def freeze(accepted_plan):
    """Freeze metadata and sources before any saved prediction is decoded."""
    require(not OUT.exists(), 'Existing diagnostic attempt cannot be reopened')
    accepted_plan = Path(accepted_plan).resolve()
    require(accepted_plan.is_file(), 'Accepted plan must exist')
    require(sha(ORIGINAL / 'plan.json') == PILOT_PLAN_SHA, 'Pilot plan differs')
    prior = read(ORIGINAL / 'plan.json')
    audit = read(ORIGINAL / 'INDEPENDENT_AUDIT.json')
    require(audit['status'] == 'PASS' and read(ORIGINAL / 'COMPLETE.json')['audit_sha256'] == sha(ORIGINAL / 'INDEPENDENT_AUDIT.json'), 'Pilot not audited')
    sources = [Path(__file__).resolve(), Path(__file__).with_name('test_diagnostic.py'),
               WORK / 'diagnostic_check.py', ROOT / 'research/world-model-planning-v1/world.py']
    require(all(p.is_file() for p in sources), 'Missing diagnostic/checker source')
    originals = inventory(ORIGINAL)
    originals.update(inventory(ROOT / 'outputs/world-model-planning/REPAIR_V1'))
    plan = {
        'schema': 'direct11-saved-prediction-diagnostic-v1',
        'status': 'FROZEN_POST_HOC_DIAGNOSTIC', 'at_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'direct_11', 'accepted_plan_path': str(accepted_plan.relative_to(ROOT)),
        'accepted_plan_sha256': sha(accepted_plan), 'pilot_plan_sha256': PILOT_PLAN_SHA,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
        'original_inventory': originals,
        'cases': prior['planning_cases'], 'states': prior['states'],
        'heldout_rows': prior['test'], 'training_rows': prior['train'],
        'execution_arms': ['soft_feedback', 'soft_blind', 'argmax_feedback', 'argmax_blind'],
        'repair_arms': [f'singleton_{r:03d}' for r in prior['test']] + ['joint_all24'],
        'horizon': 16, 'episode_cap': 40, 'action_tie_tolerance': 1e-12,
        'action_tie_order': ['N', 'E', 'S', 'W'], 'successor_tie_rule': 'lowest state index',
        'controller': 'One fixed H16 policy per model distribution and goal. Feedback indexes it by true state. Blind indexes it by internal state, initialized at the true start, then advances by lowest-index argmax of the selected predictive row. Internal-goal actions remain those returned by the existing planner; no internal-goal stop.',
        'termination': 'Test true goal first; then stop on repeated (true_state,internal_state), or after40 actions. Feedback internal state always equals the true state, so this exactly reproduces the original true-state-cycle rule.',
        'repair': 'Replace the entire declared state/action distribution by the exact oracle one-hot. Other rows unchanged. Every24 singleton and joint24 use native soft feedback, all600 cases; no retraining.',
        'native_census': 'All120 unique rows split by whether executed at least once in soft_feedback; also visit-weighted counts/proper losses over every executed action. BFS-optimal-set agreement both all600 initial decisions and every executed decision. Proper losses: sum Brier and capped NLL with exact-zero counts, no smoothing.',
        'counts': {'execution_episodes': 2400, 'repair_episodes': 15000, 'total_episodes': 17400},
        'limits': ['Outcome-exposed diagnostic on one selected successful seed and one old map; no new confirmation.',
                   'Blind execution is a declared alternative controller, not a unique feedback mediation estimand.',
                   'Argmax replacement changes full transition distributions; no calibration or single-feature claim.',
                   'Singleton harm removed by joint repair supports an interaction in this model/planner, not a unique learned compensation mechanism.',
                   'Separate checker uses another codepath written by the same author; source review by another agent is reported separately.'],
        'resources': {'seconds': 300, 'max_output_bytes': 64 * 2**20},
        'model_calls': 0, 'fits': 0, 'provider_calls': 0, 'new_spending_usd': 0,
    }
    OUT.mkdir(parents=True)
    (OUT / 'sources').mkdir()
    for i, p in enumerate(sources):
        (OUT / 'sources' / f'{i}-{p.name}').write_bytes(p.read_bytes())
    write(OUT / 'plan.json', plan)
    return {'plan_sha256': sha(OUT / 'plan.json'), 'output': str(OUT)}


def primary_policy(P, states, goal):
    path = ROOT / 'research/world-model-planning-v1/world.py'
    require(sha(path) == WORLD_SHA, 'Changed original planner')
    spec = importlib.util.spec_from_file_location('frozen_pilot_world', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy(P, states, goal, 16)


def simulate(table, states, P, pi, ties, case, blind=False, cap=40):
    """True state is evaluated externally; only the declared state indexes pi."""
    true = internal = int(case['start'])
    seen = {(true, internal)}
    tr, intr, actions, tie_counts = [true], [internal], [], []
    reason = None
    for _ in range(cap):
        if list(states[true][:2]) == case['goal']:
            reason = 'goal'
            break
        action = int(pi[internal]); actions.append(action); tie_counts.append(int(ties[internal]))
        nxt = int(table[true, action])
        internal = int(np.argmax(P[internal, action])) if blind else nxt
        true = nxt; tr.append(true); intr.append(internal)
        if list(states[true][:2]) == case['goal']:
            reason = 'goal'
            break
        pair = (true, internal)
        if pair in seen:
            reason = 'joint_cycle'
            break
        seen.add(pair)
    if reason is None:
        reason = 'cap'
    return {**case, 'true_states': tr, 'internal_states': intr, 'actions': actions,
            'tie_counts': tie_counts, 'success': reason == 'goal', 'stop_reason': reason,
            'steps': len(actions)}


def distances(table, states, goal):
    reverse = [[] for _ in states]
    for s, row in enumerate(table):
        for t in set(map(int, row)):
            reverse[t].append(s)
    d = [None] * len(states); q = deque()
    for i, s in enumerate(states):
        if list(s[:2]) == list(goal):
            d[i] = 0; q.append(i)
    while q:
        t = q.popleft()
        for s in reverse[t]:
            if d[s] is None:
                d[s] = d[t] + 1; q.append(s)
    require(all(v is not None for v in d), 'Unreachable goal')
    return d


def loss_summary(P, table, weights):
    flat = P.reshape(-1, P.shape[-1]); y = table.reshape(-1)
    w = np.asarray(weights, dtype=np.int64)
    total = int(w.sum())
    maxima = flat.max(axis=1); nties = (flat == maxima[:, None]).sum(axis=1)
    correct = (flat.argmax(axis=1) == y) & (nties == 1)
    truep = flat[np.arange(len(y)), y]
    brier = (flat * flat).sum(axis=1) - 2 * truep + 1
    nll = -np.log(np.maximum(truep, np.finfo(np.float64).tiny))
    return {'n': total, 'correct': int(np.dot(w, correct.astype(np.int64))),
            'wrong': int(np.dot(w, ((~correct) & (nties == 1)).astype(np.int64))),
            'ties': int(np.dot(w, (nties > 1).astype(np.int64))),
            'mean_true_probability': float(np.dot(w, truep) / total) if total else None,
            'mean_sum_brier': float(np.dot(w, brier) / total) if total else None,
            'capped_nll': float(np.dot(w, nll) / total) if total else None,
            'zero_true_probability': int(np.dot(w, (truep == 0).astype(np.int64)))}


def census(P, table, states, native, train, test):
    visits = np.zeros(120, dtype=np.int64)
    optimal = total = initial_optimal = wrong_rows_optimal = wrong_rows_visits = 0
    ds = {tuple(r['goal']): distances(table, states, r['goal']) for r in native}
    for row in native:
        d = ds[tuple(row['goal'])]
        for j, (s, action) in enumerate(zip(row['true_states'], row['actions'])):
            visits[4*s+action] += 1
            ok = 1+d[int(table[s, action])] == d[s]
            optimal += int(ok); total += 1
            if j == 0: initial_optimal += int(ok)
            winners = np.flatnonzero(P[s, action] == P[s, action].max())
            incorrect = len(winners) != 1 or int(winners[0]) != int(table[s, action])
            wrong_rows_visits += int(incorrect)
            wrong_rows_optimal += int(incorrect and ok)
    groups = {'all_unique': loss_summary(P, table, np.ones(120, dtype=int)),
              'on_policy_unique': loss_summary(P, table, visits > 0),
              'off_policy_unique': loss_summary(P, table, visits == 0),
              'execution_weighted': loss_summary(P, table, visits)}
    for label, rows in [('train', train), ('heldout', test)]:
        mask = np.zeros(120, dtype=int); mask[rows] = 1
        groups[label+'_unique'] = loss_summary(P, table, mask)
        groups[label+'_execution_weighted'] = loss_summary(P, table, visits*mask)
    details = []
    for row in range(120):
        s, a = divmod(row, 4); p = P[s, a]; winners = np.flatnonzero(p == p.max()).tolist()
        target = int(table[s, a]); targetp = float(p[target])
        details.append({'row': row, 'state': s, 'action': a, 'split': 'train' if row in train else 'heldout',
                        'executed_visits': int(visits[row]), 'target': target, 'predicted_set': winners,
                        'strict_correct': winners == [target], 'true_probability': targetp,
                        'sum_brier': float(np.dot(p, p)-2*targetp+1),
                        'capped_nll': float(-np.log(max(targetp, np.finfo(np.float64).tiny)))})
    return {'groups': groups, 'rows': details,
            'bfs_optimal_action': {'initial_correct': initial_optimal, 'initial_n': len(native),
                                   'executed_correct': optimal, 'executed_n': total,
                                   'incorrect_prediction_executed_n': wrong_rows_visits,
                                   'incorrect_prediction_bfs_optimal': wrong_rows_optimal}}


def summarize(rows, baseline):
    ok = [r for r in rows if r['success']]
    return {'n': len(rows), 'successes': len(ok),
            'joint_cycles': sum(r['stop_reason'] == 'joint_cycle' for r in rows),
            'caps': sum(r['stop_reason'] == 'cap' for r in rows),
            'steps_all': sum(r['steps'] for r in rows),
            'mean_steps_success': sum(r['steps'] for r in ok)/len(ok) if ok else None,
            'mean_excess_steps_success': sum(r['steps']-r['optimal_steps'] for r in ok)/len(ok) if ok else None,
            'paired_vs_soft_feedback': {
                'success_to_failure': sum(a['success'] and not b['success'] for a, b in zip(baseline, rows)),
                'failure_to_success': sum(not a['success'] and b['success'] for a, b in zip(baseline, rows)),
                'action_sequence_changed': sum(a['actions'] != b['actions'] for a, b in zip(baseline, rows)),
                'both_success_shorter': sum(a['success'] and b['success'] and b['steps'] < a['steps'] for a, b in zip(baseline, rows)),
                'both_success_longer': sum(a['success'] and b['success'] and b['steps'] > a['steps'] for a, b in zip(baseline, rows))},
            'failed_case_indices': [i for i, r in enumerate(rows) if not r['success']]}


def run():
    plan = read(OUT/'plan.json'); require(not (OUT/'STARTED.json').exists(), 'No retry')
    write(OUT/'STARTED.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(OUT/'plan.json')})
    began = time.monotonic()
    try:
        verify_inventory(plan['original_inventory'])
        for rel, h in plan['source_sha256'].items(): require(sha(ROOT/rel) == h, 'Source changed')
        require(sha(ROOT/plan['accepted_plan_path']) == plan['accepted_plan_sha256'], 'Accepted plan changed')
        original = read(ORIGINAL/'plan.json'); states = original['states']; table = np.asarray(original['transitions'], dtype=np.int64)
        logits = np.load(ORIGINAL/'predictions-direct_11.npy', allow_pickle=False)
        with np.load(ORIGINAL/'predictions.npz', allow_pickle=False) as bundle:
            require(logits.dtype == np.float32 and logits.shape == (30, 4, 30) and logits.tobytes() == bundle['direct_11'].tobytes(), 'Saved prediction identity')
        require(np.isfinite(logits).all(), 'Nonfinite saved predictions')
        x = logits.astype(np.float64); x -= x.max(axis=-1, keepdims=True)
        P = np.exp(x); P /= P.sum(axis=-1, keepdims=True)
        hard = np.eye(30, dtype=np.float64)[P.argmax(axis=-1)]
        np.savez_compressed(OUT/'probabilities.npz', soft=P, argmax=hard)
        conditions = [('soft_feedback', P, False), ('soft_blind', P, True),
                      ('argmax_feedback', hard, False), ('argmax_blind', hard, True)]
        for arm in plan['repair_arms']:
            rows = plan['heldout_rows'] if arm == 'joint_all24' else [int(arm.split('_')[1])]
            repaired = P.copy()
            for row in rows:
                s, a = divmod(row, 4); repaired[s, a] = 0; repaired[s, a, table[s, a]] = 1
            conditions.append((arm, repaired, False))
        outcomes = {}; policies = {}; kernel = {}
        for arm, probs, blind in conditions:
            require(time.monotonic()-began < plan['resources']['seconds'], 'Diagnostic timeout')
            goal_policies = {tuple(g): primary_policy(probs, states, g)
                             for g in sorted({tuple(c['goal']) for c in plan['cases']})}
            policies[arm] = {','.join(map(str, g)): {'actions': pi.tolist(), 'tie_counts': ties.tolist()}
                             for g, (pi, ties) in goal_policies.items()}
            rows = []
            for i, case in enumerate(plan['cases']):
                pi, ties = goal_policies[tuple(case['goal'])]
                rows.append({'case_index': i, 'arm': arm, **simulate(table, states, probs, pi, ties, case, blind)})
            outcomes[arm] = rows; kernel[arm] = probs
        baseline = outcomes['soft_feedback']
        old = [r for r in read(ORIGINAL/'episodes.json') if r['model'] == 'direct_11' and r['horizon'] == 16]
        require(len(old) == len(baseline) == 600, 'Incomplete baseline')
        for new, previous in zip(baseline, old):
            for key in ['start', 'goal', 'optimal_steps', 'actions', 'tie_counts', 'success', 'steps']:
                require(new[key] == previous[key], 'Native baseline replication: '+key)
            require(new['true_states'] == previous['states'] and (new['stop_reason'] == 'joint_cycle') == previous['cycle'], 'Native trace replication')
        with (OUT/'traces.jsonl.gz').open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as z:
                for rows in outcomes.values():
                    for row in rows:z.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())
        write(OUT/'policies.json', policies)
        summary = {'status': 'COLLECTED_PENDING_SEPARATE_CHECK', 'model': 'direct_11',
                   'plan_sha256': sha(OUT/'plan.json'), 'native_baseline_exact': True,
                   'native_census': census(P, table, states, baseline, plan['training_rows'], plan['heldout_rows']),
                   'execution_2x2': {k: summarize(outcomes[k], baseline) for k in plan['execution_arms']},
                   'repairs': {k: summarize(outcomes[k], baseline) for k in plan['repair_arms']},
                   'counts': plan['counts'], 'limits': plan['limits'],
                   'model_calls': 0, 'fits': 0, 'provider_calls': 0, 'new_spending_usd': 0}
        write(OUT/'summary.json', summary)
        verify_inventory(plan['original_inventory'])
        require(time.monotonic()-began < plan['resources']['seconds'], 'Diagnostic timeout at closure')
        require(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < plan['resources']['max_output_bytes'], 'Output cap')
        write(OUT/'COLLECTED.json', {'status': 'COLLECTED_PENDING_SEPARATE_CHECK', 'elapsed_seconds': time.monotonic()-began,
              'files_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'originals_unchanged': True, 'episodes': sum(map(len, outcomes.values()))})
        return {'status': summary['status'], 'output': str(OUT), 'episodes': 17400}
    except BaseException as exc:
        write(OUT/'STOP.json', {'status': 'STOP', 'type': type(exc).__name__, 'reason': str(exc)})
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='mode', required=True)
    fr = sub.add_parser('freeze'); fr.add_argument('--accepted-plan', type=Path, required=True)
    sub.add_parser('run'); args = ap.parse_args()
    print(json.dumps(freeze(args.accepted_plan) if args.mode == 'freeze' else run()), flush=True)
