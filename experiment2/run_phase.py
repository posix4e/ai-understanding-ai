"""Phase controller: confirmation requires an exact public preregistration lock."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from experiment2.model import load_model
from experiment2.runner import calibrate,run

ROOT=Path('outputs/experiment2')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checkpoint(seed):
    return Path('outputs/trained' if seed<3 else 'outputs/experiment2/checkpoints')/f'seed_{seed}.pt'

def verify_lock():
    lock=json.loads((ROOT/'remote_lock.json').read_text())
    for path,expected in lock['files'].items():
        if digest(path)!=expected:raise RuntimeError('Frozen file changed: '+path)
    remote=json.loads(subprocess.check_output(['gh','api',f"repos/posix4e/ai-understanding-ai/git/commits/{lock['commit']}"]))
    if remote['sha']!=lock['commit']:raise RuntimeError('Remote commit mismatch')
    return lock['commit']

def mean_interval(values):
    x=np.asarray(values,dtype=np.float64);mu=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    return {'mean':mu,'se':se,'ci95':[mu-1.96*se,mu+1.96*se]}

def wilson(values):
    x=np.asarray(values);p=float(x.mean());n=len(x);z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'mean':p,'ci95':[float(center-radius),float(center+radius)]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['discovery','confirmatory'],required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    commit=verify_lock() if args.phase=='confirmatory' else None
    folder=ROOT/args.phase
    if folder.exists():raise RuntimeError('Refusing to overwrite '+str(folder))
    folder.mkdir()
    (folder/'start.json').write_text(json.dumps({'phase':args.phase,'utc':datetime.now(timezone.utc).isoformat(),
                                              'preregistration_commit':commit},indent=2)+'\n')
    inputs=np.load(ROOT/f'{args.phase}_inputs.npz')
    arrays={key:inputs[key] for key in inputs.files}
    condition_sets=json.loads(Path('experiment2/conditions.json').read_text())
    conditions=condition_sets[args.phase]
    summaries={}
    (ROOT/'means').mkdir(exist_ok=True)
    for seed in range(6):
        model,_=load_model(checkpoint(seed))
        meanfile=ROOT/'means'/f'seed_{seed}.pt'
        if args.phase=='discovery':
            if meanfile.exists():raise RuntimeError('Refusing to overwrite calibration means')
            means=calibrate(model,np.load(ROOT/'calibration_inputs.npz')['tokens'])
            torch.save(means,meanfile)
        else:
            means=torch.load(meanfile,weights_only=True,map_location='cpu')
        raw=run(model,arrays,conditions,means)
        flat={f'{condition}/{metric}':np.asarray(values) for condition,metrics in raw.items() for metric,values in metrics.items()}
        if not all(np.isfinite(v).all() for v in flat.values()):raise RuntimeError('Nonfinite results')
        np.savez_compressed(folder/f'seed_{seed}.npz',**flat)
        summaries[str(seed)]={'checkpoint':str(checkpoint(seed)),'checkpoint_sha256':digest(checkpoint(seed)),
            'n':len(arrays['tokens']),'conditions':{condition:{metric:wilson(values) if metric=='accuracy' else mean_interval(values)
                for metric,values in metrics.items()} for condition,metrics in raw.items()},
            'noop_max_probability_error':max(float(np.max(np.abs(flat[f'L{layer}_noop/target_probability']-flat['clean/target_probability']))) for layer in [1,2])}
        print(json.dumps({'phase':args.phase,'seed':seed,'accuracy':summaries[str(seed)]['conditions']['clean']['accuracy']['mean']}),flush=True)
    (folder/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
