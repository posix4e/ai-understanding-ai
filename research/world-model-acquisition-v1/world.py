"""Deterministic map panel and evaluator. Never imported by selection.py."""
from collections import deque
import hashlib
import numpy as np

ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def transition(state, action, world):
    x, y, key = state
    dx, dy = ACTIONS[action]
    p = (x + dx, y + dy)
    if not (0 <= p[0] < 5 and 0 <= p[1] < 5) or list(p) in world['walls'] or (list(p) == world['door'] and not key):
        return tuple(state)
    return (*p, int(key or list(p) == world['key']))

def panel():
    layouts = []
    for d in range(5):
        for x in range(2):
            for y in range(5):
                if (x, y) == (0, 0) or (x, y, d) == (0, 4, 2):
                    continue
                layouts.append((sha(f'wm-active-v1:20261006:{d}:{x}:{y}'), d, x, y))
    maps = []
    for i, (h, d, x, y) in enumerate(sorted(layouts)[:12]):
        def rotate(p):
            a, b = p
            for _ in range(i % 4):
                a, b = 4-b, a
            return [a, b]
        world = dict(start=rotate((0, 0))+[0], key=rotate((x, y)), door=rotate((2, d)),
                     walls=sorted(rotate((2, j)) for j in range(5) if j != d))
        seen = {tuple(world['start'])}
        queue = deque(seen)
        while queue:
            s = queue.popleft()
            for a in range(4):
                t = transition(s, a, world)
                if t not in seen:
                    seen.add(t)
                    queue.append(t)
        states = sorted(seen)
        index = {s: j for j, s in enumerate(states)}
        table = [[index[transition(s, a, world)] for a in range(4)] for s in states]
        name = f'map_{i:02d}'
        splits = dict(train=[], query=[], audit=[])
        for a in range(4):
            rows = sorted(range(a, 120, 4), key=lambda r: sha(f'wm-active-v1:split:{name}:{r}'))
            for split, subset in zip(splits, (rows[:18], rows[18:24], rows[24:])):
                splits[split].extend(subset)
        splits = {k: sorted(v) for k, v in splits.items()}
        goals = sorted({s[:2] for s in states})
        assert len(states) == 30 and len(goals) == 21
        maps.append(dict(id=name, index=i, phase='engineering' if i < 4 else 'evaluation',
                         layout_hash=h, base_layout=dict(door_y=d, key_x=x, key_y=y),
                         clockwise_quarter_turns=i % 4, world=world, states=states,
                         transitions=table, goals=goals, splits=splits))
    return maps

def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.shape != (30, 4, 30) or not np.isfinite(x).all():
        raise ValueError('Invalid logits')
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)

def planner(P, states, goal):
    mask = np.array([tuple(s[:2]) == tuple(goal) for s in states])
    v = np.array([abs(s[0]-goal[0])+abs(s[1]-goal[1]) for s in states], dtype=np.float64)
    for _ in range(16):
        q = 1 + np.einsum('san,n->sa', P, v, optimize=False)
        v = q.min(axis=1)
        v[mask] = 0
    ties = q <= q.min(axis=1, keepdims=True)+1e-12
    return ties.argmax(axis=1), ties.sum(axis=1)

def distances(table, states, goal):
    reverse = [[] for _ in states]
    for s, row in enumerate(table):
        for t in set(row):
            reverse[t].append(s)
    d = [None]*len(states)
    queue = deque()
    for s, state in enumerate(states):
        if tuple(state[:2]) == tuple(goal):
            d[s] = 0
            queue.append(s)
    while queue:
        t = queue.popleft()
        for s in reverse[t]:
            if d[s] is None:
                d[s] = d[t]+1
                queue.append(s)
    return d

def episode(table, states, start, goal, pi, ties):
    s = start
    seen = {s}
    trace, actions, counts = [s], [], []
    success, cycle = False, False
    for _ in range(40):
        if tuple(states[s][:2]) == tuple(goal):
            success = True
            break
        a = int(pi[s])
        actions.append(a)
        counts.append(int(ties[s]))
        s = int(table[s][a])
        trace.append(s)
        if tuple(states[s][:2]) == tuple(goal):
            success = True
            break
        if s in seen:
            cycle = True
            break
        seen.add(s)
    return dict(start=start, goal=list(goal), states=trace, actions=actions, tie_counts=counts,
                success=success, cycle=cycle, steps=len(actions))

def evaluate(P, m):
    rows = []
    for goal in m['goals']:
        pi, ties = planner(P, m['states'], goal)
        ds = distances(m['transitions'], m['states'], goal)
        for start, state in enumerate(m['states']):
            if tuple(state[:2]) != tuple(goal):
                assert ds[start] is not None
                rows.append(dict(optimal_steps=ds[start], **episode(m['transitions'], m['states'], start, goal, pi, ties)))
    assert len(rows) == 600
    good = [r for r in rows if r['success']]
    summary = dict(n=600, successes=len(good), success_rate=len(good)/600,
                   cycles=sum(r['cycle'] for r in rows),
                   mean_steps_success=float(np.mean([r['steps'] for r in good])) if good else None,
                   mean_excess_steps_success=float(np.mean([r['steps']-r['optimal_steps'] for r in good])) if good else None)
    return summary, rows

def prediction_metrics(P, m):
    ids = m['splits']['audit']
    p = P.reshape(120, 30)[ids]
    targets = np.asarray(m['transitions']).reshape(-1)[ids]
    true = p[np.arange(24), targets]
    ties = (p == p.max(axis=1, keepdims=True)).sum(axis=1)
    return dict(n=24, correct=int(((p.argmax(axis=1) == targets) & (ties == 1)).sum()),
                capped_nll=float(-np.log(np.maximum(true, np.finfo(float).tiny)).mean()),
                zero_true_probability=int((true == 0).sum()), mean_true_probability=float(true.mean()))
