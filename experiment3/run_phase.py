"""The only E3 trained-model entrypoint; exact public freeze precedes forwards."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment3.preregister import ROOT, checkpoint, digest, verify_local_lock
from experiment3.runner import run
from model import load_model


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = verify_local_lock()
    folder = ROOT / 'confirmatory'
    if folder.exists():
        raise RuntimeError('Refusing to overwrite or retrospectively restart confirmation')
    protocol = json.loads(Path('experiment3/protocol.json').read_text())
    conditions = json.loads(Path('experiment3/conditions.json').read_text())
    with np.load(ROOT / 'inputs.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if len(arrays['tokens']) != protocol['n']:
        raise RuntimeError('Input count mismatch')
    folder.mkdir()
    (folder / 'start.json').write_text(json.dumps({
        'experiment': 3, 'utc': datetime.now(timezone.utc).isoformat(),
        'preregistration_commit': lock['commit'],
        'verified_blob_count': lock['verified_blob_count'],
        'execution': 'Local CPU, four torch threads; no training or inference API'
    }, indent=2) + '\n')
    summaries = {}
    for seed in protocol['model_seeds']:
        model, _ = load_model(checkpoint(seed))
        output = run(model, arrays, conditions)
        raw = {f'{name}/{metric}': np.asarray(values)
               for name, metrics in output.items() for metric, values in metrics.items()}
        if not all(np.isfinite(values).all() for values in raw.values()):
            raise RuntimeError('Nonfinite outputs; retain partial run for diagnosis')
        np.savez_compressed(folder / f'seed_{seed}.npz', **raw)
        summaries[str(seed)] = {
            'checkpoint': str(checkpoint(seed)), 'checkpoint_sha256': digest(checkpoint(seed)),
            'n': len(arrays['tokens']),
            'conditions': {name: {metric: np.asarray(values).mean(axis=0).tolist()
                                  for metric, values in metrics.items()}
                           for name, metrics in output.items()},
            'noop_max_probability_error': float(np.max(np.abs(
                raw['original_native/original_probability'] - raw['original_noop/original_probability'])))
        }
        print(json.dumps({'seed': seed, 'completed_conditions': len(conditions)}), flush=True)
    (folder / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    (folder / 'finish.json').write_text(json.dumps({
        'utc': datetime.now(timezone.utc).isoformat(), 'seeds': protocol['model_seeds']
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
