"""Only E5 trained-model entrypoint. Verify public freeze before any forward."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiment5.conditions import generate_conditions
from experiment5.preregister import ROOT, checkpoint, digest, verify_local_lock
from experiment5.runner import run_layout
from model import load_model


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = verify_local_lock()
    folder = ROOT / 'confirmatory'
    protocol = json.loads(Path('experiment5/protocol.json').read_text())
    grid = json.loads(Path('experiment5/conditions.json').read_text())
    if grid != generate_conditions():
        raise RuntimeError('Stored and generated condition grids differ')
    with np.load(ROOT / 'inputs.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if len(arrays['tokens']) != protocol['n']:
        raise RuntimeError('Input count mismatch')
    if folder.exists():
        if not args.resume or (folder / 'finish.json').exists():
            raise RuntimeError('Refusing to replace existing or completed confirmation')
        start = json.loads((folder / 'start.json').read_text())
        if start['preregistration_commit'] != lock['commit']:
            raise RuntimeError('Resume commit differs')
        with (folder / 'resume_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': now(), 'commit': lock['commit']}) + '\n')
    else:
        if args.resume:
            raise RuntimeError('No interrupted run to resume')
        folder.mkdir()
        save_json(folder / 'start.json', {'experiment': 5, 'utc': now(),
                  'preregistration_commit': lock['commit'],
                  'verified_blob_count': lock['verified_blob_count'],
                  'execution': 'Local CPU; four threads; fixed checkpoint cohort; no fitting or inference API'})
    panels = [('references', [grid['references']]), ('primary', grid['primary']),
              ('edge_panel', [grid['edge_panel']]), ('boundary', grid['boundary'])]
    for seed in protocol['model_seeds']:
        seed_folder = folder / f'seed_{seed}'
        seed_folder.mkdir(exist_ok=True)
        index_path = seed_folder / 'index.json'
        ledger = (json.loads(index_path.read_text()) if index_path.exists() else
                  {'seed': seed, 'checkpoint': str(checkpoint(seed)),
                   'checkpoint_sha256': digest(checkpoint(seed)), 'panels': {}})
        if ledger['checkpoint_sha256'] != digest(checkpoint(seed)):
            raise RuntimeError('Resume checkpoint mismatch')
        model, _ = load_model(checkpoint(seed))
        for panel, layouts in panels:
            panel_folder = seed_folder / panel
            panel_folder.mkdir(exist_ok=True)
            n = protocol['boundary_n'] if panel == 'boundary' else protocol['n']
            panel_arrays = {k: v[:n] for k, v in arrays.items()}
            entries = ledger['panels'].setdefault(panel, {'shards': []})['shards']
            step = protocol['layouts_per_shard']
            expected_paths = set()
            for shard, begin in enumerate(range(0, len(layouts), step)):
                selected = layouts[begin:begin+step]
                path = panel_folder / f'shard_{shard:05d}.npz'
                expected_paths.add(str(path))
                matches = [entry for entry in entries if entry['path'] == str(path)]
                if matches:
                    if len(matches) != 1 or not path.exists() or digest(path) != matches[0]['sha256']:
                        raise RuntimeError('Previously saved shard failed verification')
                    if matches[0]['layout_ids'] != [layout['id'] for layout in selected]:
                        raise RuntimeError('Saved shard layout coverage differs')
                    continue
                if path.exists():
                    raise RuntimeError('Unindexed shard retained; manual integrity review needed: ' + str(path))
                started = time.perf_counter()
                raw = {}
                for layout in selected:
                    result = run_layout(model, panel_arrays, layout, batch_size=protocol['batch_size'])
                    raw.update({f'{layout["id"]}/{condition}/{metric}': values
                                for condition, metrics in result.items() for metric, values in metrics.items()})
                if not all(np.isfinite(values).all() for values in raw.values()):
                    raise RuntimeError('Nonfinite output; retain prior shards and abort')
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **raw)
                if path.stat().st_size >= 95_000_000:
                    raise RuntimeError('Shard exceeds public-file size budget; preserve and abort')
                entry = {'path': str(path), 'sha256': digest(path),
                         'layout_ids': [layout['id'] for layout in selected], 'n': n,
                         'condition_count': sum(len(layout['conditions']) for layout in selected),
                         'completed_at_utc': now(), 'elapsed_seconds': time.perf_counter()-started}
                entries.append(entry)
                save_json(index_path, ledger)
                print(json.dumps({'seed': seed, 'panel': panel,
                                  'completed_layouts': min(begin+step, len(layouts)),
                                  'total_layouts': len(layouts), 'seconds': entry['elapsed_seconds']}), flush=True)
            if {e['path'] for e in entries} != expected_paths:
                raise RuntimeError('Unexpected or incomplete shard index')
        ledger['finished_at_utc'] = now()
        save_json(index_path, ledger)
    save_json(folder / 'finish.json', {'utc': now(), 'seeds': protocol['model_seeds'],
                                      'condition_counts': grid['counts']})


if __name__ == '__main__':
    main()
