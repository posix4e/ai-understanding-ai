"""Discovery-only prediction baselines, fixed before unseen interventions run."""
import numpy as np

EPS=1e-6

def logit(p):
    p=np.clip(p,EPS,1-EPS);return np.log(p/(1-p))

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-50,50)))

def predictions(summary,conditions):
    p={c:v['target_probability']['mean'] for c,v in summary['conditions'].items()}
    clean=p['clean']
    result={name:{} for name in ['zero_effect','additive','multiplicative','count_linear','count_logit']}
    for condition in conditions:
        name=condition['id']
        if name in p:
            for baseline in result:result[baseline][name]=p[name]
            # Zero effect really does predict the clean outcome even on familiar sites.
            result['zero_effect'][name]=clean
            continue
        single=[];layer_preds={'count_linear':[],'count_logit':[]}
        for op in condition['ops']:
            layer=op['layer']+1
            kind=op.get('corruption',op['kind'])
            heads=[h for h in range(4) if h not in op['heads']] if op['kind']=='rescue' else op['heads']
            values=np.array([p[f'L{layer}_{kind}_h{h}'] for h in heads]);single.extend(values)
            all_heads=p[f'L{layer}_{kind}_all'];k=len(heads);weight=(k-1)/3
            layer_preds['count_linear'].append(float((1-weight)*values.mean()+weight*all_heads))
            layer_preds['count_logit'].append(float(sigmoid((1-weight)*logit(values).mean()+weight*logit(all_heads))))
        result['zero_effect'][name]=clean
        result['additive'][name]=float(np.clip(clean-sum(clean-v for v in single),0,1))
        result['multiplicative'][name]=float(np.clip(clean*np.prod(np.array(single)/max(clean,EPS)),0,1))
        for baseline,values in layer_preds.items():
            result[baseline][name]=float(np.clip(clean*np.prod(np.array(values)/max(clean,EPS)),0,1))
    return result

def case_predictions(raw,conditions):
    """Match the AI predictor's access to paired per-case discovery probabilities.

    Added after discovery and before registration; no novel outcomes are used.
    """
    p={key.split('/')[0]:np.asarray(raw[key],dtype=np.float64) for key in raw.files if key.endswith('/target_probability')}
    clean=p['clean'];clean_logit=logit(clean)
    out={name:{} for name in ['case_logit_additive','case_logit_count']}
    for condition in conditions:
        name=condition['id']
        if name in p:
            for method in out:out[method][name]=float(p[name].mean())
            continue
        additive=clean_logit.copy();count=clean_logit.copy()
        for op in condition['ops']:
            layer=op['layer']+1;kind=op.get('corruption',op['kind'])
            heads=[h for h in range(4) if h not in op['heads']] if op['kind']=='rescue' else op['heads']
            singleton=np.stack([logit(p[f'L{layer}_{kind}_h{h}']) for h in heads])
            additive+=np.sum(singleton-clean_logit,axis=0)
            weight=(len(heads)-1)/3
            effective=(1-weight)*singleton.mean(0)+weight*logit(p[f'L{layer}_{kind}_all'])
            count+=effective-clean_logit
        out['case_logit_additive'][name]=float(sigmoid(additive).mean())
        out['case_logit_count'][name]=float(sigmoid(count).mean())
    return out
