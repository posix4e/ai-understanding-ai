"""Fresh dictionaries, with exact historical exclusion and no model forwards."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_holdout import encode
from experiment4.data import dictionary_variants, forbidden_inputs as old_forbidden
from model import generate_batch
from prepare_holdout import identity

ROOT = Path('outputs/experiment5')
N = 1024
INPUT_SEED = 10100001
SEEDS = list(range(6, 12))


def token_bag(row):
    return tuple(sorted(map(int, row[:8:2]))), tuple(sorted(map(int, row[1:8:2])))


def forbidden_inputs():
    forbidden = old_forbidden()
    with np.load('outputs/experiment4/inputs.npz') as data:
        for array in (data['tokens'], data['tokens'][:, [0, 2, 4, 6, 1, 3, 5, 7, 8]]):
            forbidden.update(map(int, encode(torch.from_numpy(array))))
    for seed in SEEDS:
        meta = json.loads((ROOT / f'checkpoints/seed_{seed}.json').read_text())
        if (meta['seed'], meta['training_data_seed'], meta['steps'], meta['batch_size']) != (seed, seed+1000, 2000, 128):
            raise RuntimeError('Unexpected training recipe')
        generator = torch.Generator().manual_seed(seed + 1000)
        for _ in range(2000):
            tokens, _, _ = generate_batch(128, generator)
            forbidden.update(map(int, encode(tokens)))
    return forbidden


def historical_non_native_bags():
    bags = set()
    for phase in (3, 4):
        with np.load(f'outputs/experiment{phase}/inputs.npz') as data:
            bags.update(token_bag(row) for row in data['tokens'])
    return bags


def main():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'inputs.npz'
    if output.exists():
        raise RuntimeError('Refusing to replace E5 inputs')
    forbidden = forbidden_inputs()
    bags = historical_non_native_bags()
    generator = torch.Generator().manual_seed(INPUT_SEED)
    rows, seen, rejected = [], set(), []
    candidates = 0
    while len(rows) < N:
        tokens, _, meta = generate_batch(128, generator)
        for i in range(len(tokens)):
            q = len(rows) % 4
            row = tokens[i].clone()
            row[-1] = meta['keys'][i, q]
            key = tuple(sorted(zip(map(int, row[:8:2]), map(int, row[1:8:2]))))
            variants = dictionary_variants(row.numpy())
            reason = ('prior_associations_any_pair_order_or_query'
                      if any(int(code) in forbidden for code in encode(torch.from_numpy(variants))) else
                      'prior_non_native_token_bag' if token_bag(row) in bags else
                      'duplicate_dictionary' if key in seen else None)
            if reason:
                rejected.append({'candidate': candidates, 'reason': reason})
            else:
                rows.append({'tokens': row.numpy(), 'targets': int(meta['values'][i, q]),
                             'query_pair': q, 'values': meta['values'][i].numpy()})
                seen.add(key)
            candidates += 1
            if len(rows) == N:
                break
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(output, **arrays)
    audit = {'n': N, 'boundary_n': 256, 'boundary_selection': 'first 256 accepted rows; fixed before outcomes',
             'input_seed': INPUT_SEED, 'query_counts': np.bincount(arrays['query_pair'], minlength=4).tolist(),
             'boundary_query_counts': np.bincount(arrays['query_pair'][:256], minlength=4).tolist(),
             'candidate_batch_size': 128, 'candidates': candidates, 'rejected': rejected,
             'forbidden_distinct_inputs': len(forbidden), 'forbidden_sha256': identity(forbidden),
             'prior_non_native_token_bag_count': len(bags), 'unique_dictionaries': len(seen),
             'input_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'pass': True,
             'scope': 'All prior audited training/validation/E1/E2/E3/E4 inputs plus new seeds6–11 training. '
                      'Reject all96native pair-order/query variants. Additionally reject any E3/E4 key/value '
                      'token bag regardless of associations, excluding collisions with prior grouped inputs '
                      'under every possible new layout. Boundary is a declared subset, not independent data.'}
    (ROOT / 'input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'rejected'}, indent=2))


if __name__ == '__main__':
    main()
