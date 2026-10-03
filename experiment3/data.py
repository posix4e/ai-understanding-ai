"""Freeze fresh balanced dictionaries without loading or evaluating a model."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from audit_holdout import encode
from experiment2.data import base_forbidden
from model import generate_batch
from prepare_holdout import identity

ROOT = Path('outputs/experiment3')
INPUT_SEED = 7100001
N = 2048
PAIR_ORDERS = list(itertools.permutations(range(4)))


def dictionary_variants(token):
    """All 24 pair orders x four query choices for the same associations."""
    row = np.asarray(token, dtype=np.int64)
    order_indices = np.array([[j for i in order for j in (2*i, 2*i+1)]
                              for order in PAIR_ORDERS])
    variants = np.empty((96, 9), dtype=np.int64)
    variants[:, :8] = np.repeat(row[order_indices], 4, axis=0)
    variants[:, 8] = np.tile(row[:8:2], 24)
    return variants


def forbidden_inputs():
    forbidden = base_forbidden()
    for split in ('calibration', 'discovery', 'confirmatory'):
        with np.load(f'outputs/experiment2/{split}_inputs.npz') as data:
            for name in ('tokens', 'donor_tokens', 'matched_donor_tokens'):
                forbidden.update(map(int, encode(torch.from_numpy(data[name]))))
    return forbidden


def main():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'inputs.npz'
    if output.exists():
        raise RuntimeError('Refusing to replace frozen inputs')
    forbidden = forbidden_inputs()
    generator = torch.Generator().manual_seed(INPUT_SEED)
    rows, seen, rejected = [], set(), []
    candidates = 0
    while len(rows) < N:
        tokens, _, meta = generate_batch(128, generator)
        for i in range(len(tokens)):
            query = len(rows) % 4
            token = tokens[i].clone()
            token[-1] = meta['keys'][i, query]
            variants = dictionary_variants(token.numpy())
            codes = encode(torch.from_numpy(variants))
            dictionary = tuple(sorted(zip(map(int, token[:-1:2]), map(int, token[1::2]))))
            reason = ('prior_dictionary_any_pair_order_or_query'
                      if any(int(code) in forbidden for code in codes) else
                      'duplicate_dictionary' if dictionary in seen else None)
            if reason:
                rejected.append({'candidate': candidates, 'reason': reason})
            else:
                rows.append({'tokens': token.numpy(),
                             'targets': int(meta['values'][i, query]),
                             'query_pair': query,
                             'values': meta['values'][i].numpy()})
                seen.add(dictionary)
            candidates += 1
            if len(rows) == N:
                break
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(output, **arrays)
    grouped = arrays['tokens'][:, [0, 2, 4, 6, 1, 3, 5, 7, 8]]
    grouped_codes = encode(torch.from_numpy(grouped))
    collisions = sum(int(code) in forbidden for code in grouped_codes)
    if collisions:
        raise RuntimeError('Unexpected prior grouped input collision')
    audit = {
        'n': N, 'input_seed': INPUT_SEED, 'candidate_batch_size': 128,
        'query_assignment': 'accepted row index modulo 4; no model outcomes used',
        'query_counts': np.bincount(arrays['query_pair'], minlength=4).tolist(),
        'forbidden_distinct_inputs': len(forbidden), 'forbidden_sha256': identity(forbidden),
        'candidates': candidates, 'rejected': rejected,
        'unique_dictionaries': len(seen), 'grouped_prior_collisions': collisions,
        'input_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'pass': True,
        'scope': 'Exclude all 96 variants (24 pair orders x four query choices) of each '
                 'candidate association dictionary against every audited training, '
                 'validation, E1 and E2 input. Also exclude duplicate unordered '
                 'association dictionaries within E3. Reused trained checkpoints. '
                 'Each dictionary appears with one balanced query and is shared '
                 'across all conditions and seeds.'
    }
    (ROOT / 'input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
