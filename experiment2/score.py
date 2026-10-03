"""Frozen E2 scoring: family-balanced novel-condition loss and paired bootstrap."""
import json
from pathlib import Path
import numpy as np
from experiment2.baselines import predictions as baseline_predictions

ROOT=Path('outputs/experiment2')
GROUPS=['pair','triple_rescue','cross_layer']

def interval(values):return [float(x) for x in np.quantile(values,[.025,.975])]

def main():
    conditions=json.loads(Path('experiment2/conditions.json').read_text())['confirmatory']
    names=[c['id'] for c in conditions]
    masks={group:np.array([c['group']==group for c in conditions]) for group in GROUPS}
    novel=np.logical_or.reduce(list(masks.values()))
    discovery=json.loads((ROOT/'discovery/summary.json').read_text())
    ai=json.loads(Path('experiment2/predictions.json').read_text())['expected_probability']
    frozen_baselines=json.loads(Path('experiment2/baseline_predictions.json').read_text())
    sample=np.load(ROOT/'confirmatory/seed_0.npz')
    n=len(sample['clean/target_probability'])
    rng=np.random.default_rng(6200001)
    weights=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.float64)/n
    results={}
    for seed in range(6):
        s=str(seed);raw=np.load(ROOT/f'confirmatory/seed_{seed}.npz')
        y=np.stack([raw[name+'/target_probability'] for name in names],axis=1).astype(np.float64)
        if not np.isfinite(y).all():raise RuntimeError('Nonfinite results')
        mean=y.mean(0);boot=weights@y
        expected={'ai':np.array([ai[s][name] for name in names]),
                  **{b:np.array([values[name] for name in names]) for b,values in frozen_baselines[s].items()}}
        row={'n':n,'novel_conditions':int(novel.sum()),'scores':{},'conditions':{},'group_scores':{}}
        boot_losses={}
        for method,p in expected.items():
            mse=float(np.mean([np.mean((mean[mask]-p[mask])**2) for mask in masks.values()]))
            draws=np.mean(np.stack([np.mean((boot[:,mask]-p[mask])**2,axis=1) for mask in masks.values()]),axis=0)
            boot_losses[method]=draws
            row['scores'][method]={'balanced_mse':mse,'balanced_rmse':float(np.sqrt(mse)),
                                   'bootstrap_ci95_mse':interval(draws)}
            row['group_scores'][method]={g:float(np.mean((mean[mask]-p[mask])**2)) for g,mask in masks.items()}
        best=min((m for m in expected if m!='ai'),key=lambda m:row['scores'][m]['balanced_mse'])
        best_boot=np.min(np.stack([v for k,v in boot_losses.items() if k!='ai']),axis=0)
        row['ai_minus_best_baseline']={'best_point_baseline':best,
            'difference':row['scores']['ai']['balanced_mse']-row['scores'][best]['balanced_mse'],
            'bootstrap_ci95':interval(boot_losses['ai']-best_boot),
            'bootstrap_definition':'AI loss minus minimum of seven baseline losses in each paired resample'}
        err=np.abs(mean-expected['ai'])
        row['novel_within_0_10']=int((err[novel]<=.10).sum())
        row['novel_fraction_within_0_10']=float((err[novel]<=.10).mean())
        row['largest_novel_error']=float(err[novel].max())
        for i,c in enumerate(conditions):
            vals=y[:,i];se=float(vals.std(ddof=1)/np.sqrt(n))
            row['conditions'][c['id']]={'group':c['group'],'predicted':float(expected['ai'][i]),
                'observed':float(mean[i]),'ci95':[float(mean[i]-1.96*se),float(mean[i]+1.96*se)],
                'error':float(mean[i]-expected['ai'][i]),'accuracy':float(raw[c['id']+'/accuracy'].mean())}
        row['prespecified_prediction_gates']={
            'balanced_rmse_le_0_10':row['scores']['ai']['balanced_rmse']<=.10,
            'at_least_80pct_novel_within_0_10':row['novel_fraction_within_0_10']>=.8,
            'beats_best_baseline_ci95':row['ai_minus_best_baseline']['bootstrap_ci95'][1]<0}
        summary=json.loads((ROOT/'confirmatory/summary.json').read_text())[s]
        row['validity_gates']={'clean_accuracy_ge_0_95':float(raw['clean/accuracy'].mean())>=.95,
            'noop_probability_error_le_1e_6':summary['noop_max_probability_error']<=1e-6,
            'finite_probabilities':bool(np.isfinite(y).all())}
        results[s]=row
        print(json.dumps({'seed':seed,'scores':row['scores'],'gates':row['prespecified_prediction_gates']}),flush=True)
    (ROOT/'confirmatory/scores.json').write_text(json.dumps(results,indent=2)+'\n')
    global_gates={
        'all_six_valid':all(all(r['validity_gates'].values()) for r in results.values()),
        'all_six_predictively_adequate':all(r['prespecified_prediction_gates']['balanced_rmse_le_0_10'] and
             r['prespecified_prediction_gates']['at_least_80pct_novel_within_0_10'] for r in results.values()),
        'all_six_beat_best_baseline':all(r['prespecified_prediction_gates']['beats_best_baseline_ci95'] for r in results.values())}
    global_gates['full_claim_pass']=all(global_gates.values())
    (ROOT/'confirmatory/global_decisions.json').write_text(json.dumps(global_gates,indent=2)+'\n')

if __name__=='__main__':main()
