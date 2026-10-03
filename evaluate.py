"""Discovery/locked-confirmation causal interventions; all execution is local CPU."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from model import load_model, generate_batch

SITES = {
    'l1_resid_values': (0, 'resid_post', 'values'),
    'l1_resid_query': (0, 'resid_post', 'query'),
    'l2_k_values': (1, 'k', 'values'),
    'l2_q_query': (1, 'q', 'query'),
    'l2_v_values': (1, 'v', 'values'),
    'l2_k_keys': (1, 'k', 'keys'),
    'l2_q_values': (1, 'q', 'values'),
    'l1_resid_wrong_values': (0, 'resid_post', 'wrong_values'),
    'l2_k_wrong_values': (1, 'k', 'wrong_values'),
}
FAMILIES = ['key', 'value', 'matched_key', 'matched_value']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_lock():
    lock = json.loads(Path('outputs/remote_lock.json').read_text())
    for path, expected in lock['files'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Frozen file changed: {path}')
    # Read the exact public Git object again, immediately before confirmation.
    remote = json.loads(subprocess.check_output(['gh', 'api',
        f"repos/posix4e/ai-understanding-ai/git/commits/{lock['commit']}"]))
    if remote['sha'] != lock['commit']:
        raise RuntimeError('Remote commit verification failed')
    return lock['commit']

def make_donors(tokens, meta, generator):
    b = len(tokens)
    rows = torch.arange(b)
    a = meta['query_pair']
    offset = torch.randint(1, 4, (b,), generator=generator)
    other = (a + offset) % 4
    remaining = torch.stack([torch.tensor([j for j in range(4) if j not in (int(x), int(y))])
                             for x, y in zip(a, other)])
    out = {}
    for family in FAMILIES:
        x, y = (remaining[:, 0], remaining[:, 1]) if family.startswith('matched') else (a, other)
        token_offset = 0 if family.endswith('key') else 1
        donor = tokens.clone()
        donor[rows, 2*x+token_offset] = tokens[rows, 2*y+token_offset]
        donor[rows, 2*y+token_offset] = tokens[rows, 2*x+token_offset]
        out[family] = (donor, x, y)
    return out, other, remaining

def patch_tensor(recipient, donor, kind, a, b, remaining):
    mixed = recipient.clone()
    rows = torch.arange(len(mixed))
    if kind == 'query':
        mixed[:, -1] = donor[:, -1]
    else:
        if kind == 'wrong_values':
            x, y = remaining[:, 0], remaining[:, 1]
        else:
            offset = 0 if kind == 'keys' else 1
            x, y = 2*a + offset, 2*b + offset
        if kind == 'wrong_values':
            x, y = 2*x+1, 2*y+1
        mixed[rows, x] = donor[rows, x]
        mixed[rows, y] = donor[rows, y]
    return mixed

def measure(logits, original, alternative):
    probs = logits.softmax(-1)
    rows = torch.arange(len(logits))
    return torch.stack([probs[rows, alternative]-probs[rows, original],
                        (logits.argmax(-1)==alternative).float(),
                        (logits.argmax(-1)==original).float()], dim=1).numpy()

def interval(values):
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    return {'mean': mean, 'se': se, 'ci95': [mean-1.96*se, mean+1.96*se]}

@torch.no_grad()
def run(checkpoint, seed, count, batch_size):
    model, metadata = load_model(checkpoint)
    generator = torch.Generator().manual_seed(seed)
    donor_generator = torch.Generator().manual_seed(seed+10000000)
    arrays = {'tokens': [], 'query_pair': [], 'original': [], 'alternative': [], 'base': []}
    noops = {site: 0.0 for site in SITES}
    for start in range(0, count, batch_size):
        tokens, original, meta = generate_batch(min(batch_size, count-start), generator, model.config)
        donors, other, remaining = make_donors(tokens, meta, donor_generator)
        alternative = meta['values'][torch.arange(len(tokens)), other]
        base, cache = model(tokens, return_cache=True)
        for key, value in [('tokens', tokens.numpy()), ('query_pair',meta['query_pair'].numpy()),
                           ('original',original.numpy()), ('alternative',alternative.numpy()),
                           ('base',measure(base[:,-1],original,alternative))]:
            arrays[key].append(value)
        for site, (layer, component, kind) in SITES.items():
            patched = model(tokens, patch={(layer,component):(slice(None),cache[layer,component])})
            noops[site] = max(noops[site], float((patched-base).abs().max()))
        for family, (donor_tokens, a, b) in donors.items():
            logits, donor_cache = model(donor_tokens, return_cache=True)
            arrays.setdefault(family+'/unpatched', []).append(measure(logits[:,-1],original,alternative))
            # For matched donors active sites follow the irrelevant exchanged pairs;
            # wrong sites use the complementary pair (including queried association).
            unused = torch.stack([torch.tensor([j for j in range(4) if j not in (int(x),int(y))])
                                  for x,y in zip(a,b)])
            for site,(layer,component,kind) in SITES.items():
                mixed = patch_tensor(cache[layer,component],donor_cache[layer,component],kind,a,b,unused)
                patched = model(tokens, patch={(layer,component):(slice(None),mixed)})[:,-1]
                arrays.setdefault(family+'/'+site, []).append(measure(patched,original,alternative))
    arrays = {key:np.concatenate(values) for key,values in arrays.items()}
    summary = {'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
               'input_seed':seed,'n':count,'noop_max_logit_error':noops,
               'base_accuracy':interval(arrays['base'][:,2]),'conditions':{},
               'accuracy_by_query_pair':{str(i):float(arrays['base'][arrays['query_pair']==i,2].mean()) for i in range(4)}}
    for key,value in arrays.items():
        if '/' in key:
            summary['conditions'][key] = {'effect':interval(value[:,0]-arrays['base'][:,0]),
                                          'alternative_rate':interval(value[:,1]),
                                          'original_rate':interval(value[:,2])}
    return arrays, summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['discovery','confirmatory'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    commit = verify_lock() if args.phase=='confirmatory' else None
    protocol=json.loads(Path('protocol.json').read_text())
    folder=Path('outputs')/args.phase
    if folder.exists():
        raise RuntimeError(f'Refusing to overwrite {folder}')
    folder.mkdir()
    (folder/'start.json').write_text(json.dumps({'phase':args.phase,'preregistration_commit':commit,
        'utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    summaries={}
    for seed in protocol['model_seeds']:
        arrays, summary=run(Path(protocol['checkpoint_dir'])/f'seed_{seed}.pt',
                            protocol[args.phase]['input_seed'],protocol[args.phase]['n'],protocol['batch_size'])
        np.savez_compressed(folder/f'seed_{seed}.npz',**arrays)
        summaries[str(seed)]=summary
        print(json.dumps({'phase':args.phase,'seed':seed,'accuracy':summary['base_accuracy']}),flush=True)
    (folder/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':
    main()
