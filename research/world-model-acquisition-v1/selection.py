"""Pure selectors: no simulator, transition targets, filesystem or evaluation imports."""
import hashlib
import numpy as np

def key(prefix, map_id, seed, row):
    return hashlib.sha256(f'wm-active-v1:{prefix}:{map_id}:{seed}:{row}'.encode()).hexdigest()

def choose(P, states, goals, remaining, method, map_id, seed):
    remaining = sorted(remaining)
    if not remaining or len(remaining) != len(set(remaining)):
        raise ValueError('Empty or repeated candidate IDs')
    if P.shape != (30, 4, 30) or not np.isfinite(P).all() or np.any(P < 0) or not np.allclose(P.sum(-1), 1, rtol=0, atol=1e-12):
        raise ValueError('Invalid predicted probabilities')
    scores = np.zeros(120, dtype=np.float64)
    clamped, minimum = 0, 0.0
    if method == 'random':
        order = sorted(remaining, key=lambda r: key('random', map_id, seed, r))
        scores[order] = np.arange(len(order), 0, -1)
    elif method == 'uncertainty':
        logs = np.zeros_like(P)
        np.log(P, out=logs, where=P > 0)
        scores = -(P*logs).sum(axis=-1).reshape(-1)
    elif method == 'decision':
        case_count = sum(tuple(s[:2]) != tuple(g) for s in states for g in goals)
        for goal in goals:
            mask = np.array([tuple(s[:2]) == tuple(goal) for s in states])
            v = np.array([abs(s[0]-goal[0])+abs(s[1]-goal[1]) for s in states], dtype=np.float64)
            for _ in range(15):
                q = 1 + np.einsum('san,n->sa', P, v, optimize=False)
                v = q.min(axis=1)
                v[mask] = 0
            q = 1 + np.einsum('san,n->sa', P, v, optimize=False)
            pi = (q <= q.min(axis=1, keepdims=True)+1e-12).argmax(axis=1)
            distribution = (~mask).astype(float)/case_count
            weight = np.zeros(30)
            induced = P[np.arange(30), pi]
            for _ in range(40):
                weight += distribution
                distribution = distribution @ induced
                distribution[mask] = 0
            for row in remaining:
                s, a = divmod(row, 4)
                other = np.min(np.delete(q[s], a))
                gain = float(q[s].min() - np.dot(P[s, a], np.minimum(other, 1+v)))
                minimum = min(minimum, gain)
                if gain < -1e-12:
                    raise FloatingPointError(f'Decision gain below tolerance: {gain}')
                if gain < 0:
                    clamped += 1
                    gain = 0.0
                scores[row] += weight[s]*gain
    else:
        raise ValueError('Unknown selection method')
    maximum = float(max(scores[r] for r in remaining))
    tied = sorted([r for r in remaining if scores[r] >= maximum-1e-12], key=lambda r: key('tie', map_id, seed, r))
    return dict(selected=tied[0], scores={str(r): float(scores[r]) for r in remaining},
                tied=tied, clamped_negative_gains=clamped, minimum_raw_gain=minimum)
