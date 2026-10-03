"""Score frozen mean forecasts on all conditions; paired bootstrap within model."""
import json
from pathlib import Path
import numpy as np
from evaluate import FAMILIES,SITES,COMBINATIONS

def percentile(values):
    return [float(v) for v in np.quantile(values,[.025,.975])]

def main():
    protocol=json.loads(Path('protocol.json').read_text())
    predictions=json.loads(Path('predictions.json').read_text())
    discovery=json.loads(Path('outputs/discovery/summary.json').read_text())
    result={}
    conditions=[f'{family}/{site}' for family in FAMILIES for site in [*SITES,*COMBINATIONS]]
    conditions.append('cross/keyK_valueV')
    rng=np.random.default_rng(protocol['bootstrap_seed'])
    for seed in protocol['model_seeds']:
        raw=np.load(f'outputs/confirmatory/seed_{seed}.npz')
        y=np.stack([raw[c][:,0]-raw['base'][:,0] for c in conditions],axis=1)
        ai=np.array([predictions['expected_effect'][str(seed)][c] for c in conditions])
        def baseline(condition):
            family,site=condition.split('/')
            if family=='cross':
                return np.clip(sum(discovery[str(seed)]['conditions'][c]['effect']['mean']
                    for c in ['key/l2_k_values','value/l2_v_values']),-2,2)
            if site in COMBINATIONS:
                return np.clip(sum(discovery[str(seed)]['conditions'][family+'/'+s]['effect']['mean']
                                   for s in COMBINATIONS[site]),-2,2)
            return discovery[str(seed)]['conditions'][condition]['effect']['mean']
        empirical=np.array([baseline(c) for c in conditions])
        donor=np.stack([raw[c.split('/')[0]+'/unpatched'][:,0]-raw['base'][:,0] for c in conditions],axis=1)
        forecasts={'ai':ai,'discovery_mean':empirical,'zero':np.zeros(len(conditions))}
        row={'condition_count':len(conditions),'n':len(y),'scores':{}}
        # Primary forecast loss is equally weighted condition-mean squared error.
        # Individual-case MSE is secondary: constant forecasts cannot explain heterogeneity.
        for name,pred in forecasts.items():
            row['scores'][name]={'mean_mse':float(np.mean((y.mean(0)-pred)**2)),
                                 'individual_mse':float(np.mean((y-pred)**2))}
        row['scores']['full_donor']={'mean_mse':float(np.mean((y.mean(0)-donor.mean(0))**2)),
                                      'individual_mse':float(np.mean((y-donor)**2))}
        bootstrap={name:[] for name in row['scores']}
        gap=[]
        group_gaps={'familiar':[],'novel_combination':[]}
        masks={group:np.array([((c.split('/')[1] in COMBINATIONS) or c.startswith('cross/'))==novel for c in conditions])
               for group,novel in [('familiar',False),('novel_combination',True)]}
        contrast=[]
        differences=y[:,conditions.index('key/l2_k_values')]-y[:,conditions.index('key/l2_q_query')]
        for _ in range(protocol['bootstrap_replicates']):
            indices=rng.integers(0,len(y),len(y))
            means=y[indices].mean(0)
            for name,pred in forecasts.items():
                bootstrap[name].append(float(np.mean((means-pred)**2)))
            bootstrap['full_donor'].append(float(np.mean((means-donor[indices].mean(0))**2)))
            gap.append(bootstrap['ai'][-1]-bootstrap['discovery_mean'][-1])
            for group,mask in masks.items():
                group_gaps[group].append(float(np.mean((means[mask]-ai[mask])**2)-np.mean((means[mask]-empirical[mask])**2)))
            contrast.append(float(differences[indices].mean()))
        for name in row['scores']:
            row['scores'][name]['bootstrap_ci95']=percentile(bootstrap[name])
        row['ai_minus_discovery_mean_mse']={'difference':row['scores']['ai']['mean_mse']-row['scores']['discovery_mean']['mean_mse'],
                                           'bootstrap_ci95':percentile(gap)}
        row['primary_routing_contrast']={'mean':float(differences.mean()),'bootstrap_ci95':percentile(contrast)}
        row['forecast_errors']={condition:{'predicted':float(ai[i]),'observed':float(y[:,i].mean()),
                                           'absolute_error':float(abs(ai[i]-y[:,i].mean()))}
                                for i,condition in enumerate(conditions)}
        row['within_0_15']=sum(v['absolute_error']<=.15 for v in row['forecast_errors'].values())
        row['by_intervention_group']={}
        for group,mask in masks.items():
            row['by_intervention_group'][group]={name:float(np.mean((y.mean(0)[mask]-pred[mask])**2))
                for name,pred in {**forecasts,'full_donor':donor.mean(0)}.items()}
            row['by_intervention_group'][group]['ai_minus_discovery_bootstrap_ci95']=percentile(group_gaps[group])
        result[str(seed)]=row
    Path('outputs/confirmatory/scores.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:{'scores':r['scores'],'primary_routing_contrast':r['primary_routing_contrast'],
                        'within_0_15':r['within_0_15']} for s,r in result.items()},indent=2))

if __name__=='__main__':
    main()
