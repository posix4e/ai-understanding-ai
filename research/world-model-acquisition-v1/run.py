"""Bounded local acquisition study; grade only after a phase finishes acquisition."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import resource
import shutil
import sys
import time
import numpy as np
import torch
from models import create, features
from selection import choose
from world import panel, probabilities, evaluate, prediction_metrics

HERE = Path(__file__).resolve().parent
SEEDS = (11, 29, 47)
ARMS = ('random', 'uncertainty', 'decision', 'replay')
SOURCE_NAMES = ('world.py', 'selection.py', 'models.py', 'run.py', 'launch.py', 'audit.py')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def utc():
    return datetime.now(timezone.utc).isoformat()

def sources():
    return {name: digest(HERE/name) for name in SOURCE_NAMES}

def prepare(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Refuse to recreate a study')
    maps = panel()
    controls = {}
    for m in maps:
        metrics, _ = evaluate(np.eye(30)[np.array(m['transitions'])], m)
        if metrics['successes'] != 600 or metrics['mean_excess_steps_success'] != 0:
            raise RuntimeError('Exact-dynamics preflight failed')
        controls[m['id']] = metrics
    out.mkdir(parents=True)
    snapshot = out/'sources'
    snapshot.mkdir()
    for name in SOURCE_NAMES:
        shutil.copyfile(HERE/name, snapshot/name)
    shutil.copyfile(HERE/'README.md', snapshot/'README.md')
    plan = dict(schema='world-model-acquisition-v1', created_at_utc=utc(), maps=maps,
                seeds=list(SEEDS), arms=list(ARMS), budgets=[0, 1, 2, 4], rounds=4,
                training=dict(initial_steps=1500, extra_steps_per_round=250, batch_size=32,
                              original_slots=16, acquired_slots=16, learning_rate=.003,
                              optimizer='Adam', weight_decay=0, device='cpu', threads=2, dtype='float32'),
                planner=dict(horizon=16, episode_cap=40, tie_tolerance=1e-12, full_state_cycle_stop=True),
                split_hash='wm-active-v1:split:{map_id}:{row_id}',
                tie_hash='wm-active-v1:tie:{map_id}:{seed}:{row_id}',
                random_hash='wm-active-v1:random:{map_id}:{seed}:{row_id}',
                update_rng='numpy.default_rng(seed + 100000 * round); old=choice(train,(250,16)); u=random((250,16)); new=buffer[floor(u*len(buffer))]; replay buffer=train',
                source_sha256=sources(), protocol_sha256=digest(HERE/'README.md'),
                runtime=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__),
                resources=dict(timeout_seconds_per_phase=1800, rss_bytes=2*2**30, output_bytes=2**30, min_free_bytes=8*2**30),
                budget=dict(new_provider_calls=0, new_estimated_usd=0, inclusive_cap_usd=20, prior_fees_and_reserves_usd=7.6774551949),
                primary='Per evaluation map: decision-minus-random H16 success fraction at round4, mean over two architectures and three seeds; then unweighted mean over eight maps.',
                limits=['Descriptive fixed panel, not population significance or novelty.', 'Public reachable-state vocabulary; free state resets.',
                        'Equal initial evidence, label budgets, updates and objective; acquired labels may differ; not FLOP matching.', 'DI is frozen-value exact-revelation heuristic; occupancy omits pathwise cycle stopping.',
                        'No true row injection; all final distributions are neural outputs.', 'Independent audit checks evidence, not optimization gradients.'])
    dump(out/'plan.json', plan)
    dump(out/'PREFLIGHT.json', dict(status='PASS_GRAPH_PREFLIGHT', exact_controls=controls))
    print(json.dumps(dict(status='PREPARED', plan_sha256=digest(out/'plan.json'))), flush=True)

def update_batches(train, acquired, seed, round_number, replay=False):
    rng = np.random.default_rng(seed+100000*round_number)
    old = rng.choice(train, size=(250, 16), replace=True)
    u = rng.random((250, 16))
    buffer = np.asarray(train if replay else acquired)
    return np.concatenate((old, buffer[np.floor(u*len(buffer)).astype(int)]), axis=1)

def run(out, phase):
    out = Path(out)
    plan = json.loads((out/'plan.json').read_text())
    if sources() != plan['source_sha256']:
        raise RuntimeError('Scientific source differs from frozen plan')
    if phase == 'evaluation':
        audit = json.loads((out/'engineering'/'AUDIT.json').read_text())
        if audit['status'] != 'PASS' or audit['plan_sha256'] != digest(out/'plan.json') or audit['collected_sha256'] != digest(out/'engineering'/'COLLECTED.json'):
            raise RuntimeError('Engineering independent audit must pass first')
        collected = json.loads((out/'engineering'/'COLLECTED.json').read_text())
        for name, expected in collected['files_sha256'].items():
            if digest(out/'engineering'/name) != expected:
                raise RuntimeError('Engineering evidence changed after collection')
    phase_dir = out/phase
    phase_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    maps = [m for m in plan['maps'] if m['phase'] == phase]
    counter = 0
    def guard():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        if time.monotonic()-started > 1800 or rss > 2*2**30:
            raise RuntimeError('Time or RSS limit exceeded')
        if shutil.disk_usage(out).free < 8*2**30:
            raise RuntimeError('Free storage limit exceeded')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2**30:
            raise RuntimeError('Study output limit exceeded')
        return int(rss)
    def fit(model, optimizer, sf, sa, batches, observed):
        losses = []
        model.train()
        target_rows = np.asarray([[observed[int(i)] for i in batch] for batch in batches], dtype=np.int64)
        for n, (batch, y) in enumerate(zip(batches, target_rows), 1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(sa[batch], sf), torch.from_numpy(y))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if n % 250 == 0:
                guard()
                losses.append(dict(update=n, loss=float(loss.detach())))
        return dict(updates=len(batches), loss_trace=losses,
                    used_targets_sha256=hashlib.sha256(target_rows.tobytes()).hexdigest())
    def save(model, optimizer, sf, sa, directory, stem, metadata):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metadata=metadata), directory/f'{stem}.pt')
        model.eval()
        with torch.no_grad():
            logits = model(sa, sf).numpy().reshape(30, 4, 30)
        np.save(directory/f'{stem}.npy', logits, allow_pickle=False)
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite predictions')
        return logits
    try:
        guard()
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        freeze = dict(phase=phase, at_utc=utc(), plan_sha256=digest(out/'plan.json'), source_sha256=sources())
        if phase == 'evaluation':
            freeze['engineering_audit_sha256'] = digest(out/'engineering'/'AUDIT.json')
        dump(phase_dir/'FREEZE.json', freeze)
        initial_records = []
        for m in maps:
            md = phase_dir/m['id']
            md.mkdir()
            sf, sa = features(m['states'])
            targets = np.asarray(m['transitions']).reshape(-1)
            train = m['splits']['train']
            observed = {i: int(targets[i]) for i in train}
            for seed in SEEDS:
                batches = np.random.default_rng(seed).choice(train, size=(1500, 32), replace=True)
                for kind in ('direct', 'energy'):
                    directory = md/f'{kind}_{seed}'
                    directory.mkdir()
                    model, optimizer = create(kind, seed)
                    t0 = time.monotonic()
                    record = fit(model, optimizer, sf, sa, batches, observed)
                    record.update(map=m['id'], model=f'{kind}_{seed}', seconds=time.monotonic()-t0)
                    np.save(directory/'base_batches.npy', batches, allow_pickle=False)
                    save(model, optimizer, sf, sa, directory, 'base', dict(round=0, queries=[], observed=sorted(observed)))
                    dump(directory/'base_training.json', record)
                    initial_records.append(record)
                    print(json.dumps(dict(phase=phase, stage='initial_fit', map=m['id'], model=f'{kind}_{seed}')), flush=True)
        dump(phase_dir/'INITIAL_TRAINING_COMPLETE.json', dict(at_utc=utc(), fits=initial_records))
        chains = []
        for m in maps:
            sf, sa = features(m['states'])
            truth = np.asarray(m['transitions']).reshape(-1)
            for seed in SEEDS:
                for kind in ('direct', 'energy'):
                    name = f'{kind}_{seed}'
                    directory = phase_dir/m['id']/name
                    for arm in ARMS:
                        arm_dir = directory/arm
                        arm_dir.mkdir()
                        model, optimizer = create(kind, seed)
                        base = torch.load(directory/'base.pt', weights_only=True)
                        model.load_state_dict(base['model'])
                        optimizer.load_state_dict(base['optimizer'])
                        previous = directory/'base.npy'
                        acquired = []
                        observed = {i: int(truth[i]) for i in m['splits']['train']}
                        for rnd in range(1, 5):
                            guard()
                            remaining = sorted(set(m['splits']['query'])-set(acquired))
                            t0 = time.monotonic()
                            if arm != 'replay':
                                decision = choose(probabilities(np.load(previous, allow_pickle=False)), m['states'], m['goals'], remaining, arm, m['id'], seed)
                                selected = decision['selected']
                                acquired.append(selected)
                                observed[selected] = int(truth[selected])
                                decision['revealed_target'] = observed[selected]
                            else:
                                decision = dict(selected=None, revealed_target=None, scores={}, tied=[], clamped_negative_gains=0, minimum_raw_gain=0.0)
                            decision.update(round=rnd, method=arm, seed=seed, map=m['id'], remaining_before=remaining,
                                            acquired=list(acquired), prediction_sha256=digest(previous), seconds=time.monotonic()-t0)
                            dump(arm_dir/f'round_{rnd}_query.json', decision)
                            batches = update_batches(m['splits']['train'], acquired, seed, rnd, arm == 'replay')
                            np.save(arm_dir/f'round_{rnd}_batches.npy', batches, allow_pickle=False)
                            t0 = time.monotonic()
                            record = fit(model, optimizer, sf, sa, batches, observed)
                            record.update(seconds=time.monotonic()-t0, round=rnd, queries=len(acquired))
                            save(model, optimizer, sf, sa, arm_dir, f'round_{rnd}', dict(round=rnd, queries=list(acquired), observed=sorted(observed)))
                            dump(arm_dir/f'round_{rnd}_training.json', record)
                            previous = arm_dir/f'round_{rnd}.npy'
                        chains.append(dict(map=m['id'], model=name, arm=arm, queries=acquired))
                    counter += 1
                    print(json.dumps(dict(phase=phase, stage='acquisition', completed_models=counter, total_models=len(maps)*6)), flush=True)
        dump(phase_dir/'ACQUISITION_COMPLETE.json', dict(at_utc=utc(), chains=chains,
            files_sha256={str(p.relative_to(phase_dir)): digest(p) for p in sorted(phase_dir.rglob('*')) if p.is_file()}))
        # No true navigation or final-audit scoring above this barrier.
        reports = []
        for m in maps:
            report = dict(map=m['id'], phase=phase, models={})
            with gzip.open(phase_dir/m['id']/'episodes.jsonl.gz', 'wt') as traces:
                def grade(path, name, arm, budget):
                    P = probabilities(np.load(path, allow_pickle=False)) if path else np.eye(30)[np.asarray(m['transitions'])]
                    metric, rows = evaluate(P, m)
                    for row in rows:
                        traces.write(json.dumps(dict(model=name, arm=arm, budget=budget, **row), separators=(',', ':'))+'\n')
                    return dict(planning=metric, prediction_audit=prediction_metrics(P, m))
                report['oracle'] = grade(None, 'oracle', 'oracle', 0)
                for seed in SEEDS:
                    for kind in ('direct', 'energy'):
                        name = f'{kind}_{seed}'
                        directory = phase_dir/m['id']/name
                        result = dict(base=grade(directory/'base.npy', name, 'base', 0), arms={})
                        for arm in ARMS:
                            result['arms'][arm] = {str(b): grade(directory/arm/f'round_{b}.npy', name, arm, b) for b in (1, 2, 4)}
                        report['models'][name] = result
            report['primary_difference'] = float(np.mean([r['arms']['decision']['4']['planning']['success_rate']-r['arms']['random']['4']['planning']['success_rate'] for r in report['models'].values()]))
            report['query_overlap'] = []
            for seed in SEEDS:
                for kind in ('direct', 'energy'):
                    name = f'{kind}_{seed}'
                    sets = {a: set(json.loads((phase_dir/m['id']/name/a/'round_4_query.json').read_text())['acquired']) for a in ARMS if a != 'replay'}
                    report['query_overlap'].append(dict(model=name, decision_random=len(sets['decision'] & sets['random']), decision_uncertainty=len(sets['decision'] & sets['uncertainty']), uncertainty_random=len(sets['uncertainty'] & sets['random'])))
            dump(phase_dir/m['id']/'report.json', report)
            reports.append(report)
        total = dict(status='COLLECTED_PENDING_AUDIT', phase=phase, maps=reports,
                     primary_mean_difference=float(np.mean([m['primary_difference'] for m in reports])),
                     positive_maps=sum(m['primary_difference'] > 0 for m in reports),
                     seconds=time.monotonic()-started, peak_rss_bytes=guard(), new_provider_calls=0, new_estimated_usd=0)
        dump(phase_dir/'report.json', total)
        dump(phase_dir/'COLLECTED.json', dict(at_utc=utc(), status='COLLECTED_PENDING_AUDIT',
             files_sha256={str(p.relative_to(phase_dir)): digest(p) for p in sorted(phase_dir.rglob('*')) if p.is_file()}))
        print(json.dumps(dict(phase=phase, status='COLLECTED_PENDING_AUDIT', primary_mean_difference=total['primary_mean_difference'], seconds=total['seconds'])), flush=True)
    except BaseException as exc:
        dump(phase_dir/'STOP.json', dict(at_utc=utc(), type=type(exc).__name__, reason=str(exc)))
        raise

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', choices=['engineering', 'evaluation'])
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a.out)
    elif a.phase:
        run(a.out, a.phase)
    else:
        ap.error('--phase required for run')
