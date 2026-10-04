"""Score all fixed E7 exploratory conditions and contrasts from saved arrays."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from experiment6.score import stratified_weights,interval
from experiment7.conditions import CONDITIONS

METRICS=('restricted_accuracy','unrestricted_accuracy','raw_target_probability','conditional_target_probability','candidate_mass','target_logit_vs_mean_other')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/experiment7/discovery');args=p.parse_args()
    root=Path(args.root);finish=json.loads((root/'finished.json').read_text())
    raw={}
    for c in CONDITIONS:
        f=root/(c['id']+'.npz');assert hashlib.sha256(f.read_bytes()).hexdigest()==finish['files'][f.name]
        with np.load(f,allow_pickle=False) as d: raw[c['id']]={k:d[k] for k in d.files}
    ref=raw['native'];n=len(ref['query_pair']);weights=stratified_weights(ref['query_pair'],2000,10700003)
    vectors={};summaries={}
    for name,d in raw.items():
        for key in ('target_indices','target_token_ids','query_pair'): assert np.array_equal(d[key],ref[key])
        prob=d['candidate_probs'];assert prob.shape==(n,16) and np.isfinite(prob).all() and (prob>=0).all()
        logits=d['candidate_logits'];assert logits.shape==(n,16) and np.isfinite(logits).all()
        target_logits=logits[np.arange(n),d['target_indices']]
        margin=target_logits-(logits.sum(1)-target_logits)/15
        mass=prob.sum(1);assert (mass>0).all() and (mass<=1+1e-8).all()
        target=prob[np.arange(n),d['target_indices']]
        vec=dict(restricted_accuracy=(prob.argmax(1)==d['target_indices']).astype(float),
                 unrestricted_accuracy=(d['top_token_ids']==d['target_token_ids']).astype(float),
                 raw_target_probability=target,conditional_target_probability=target/mass,candidate_mass=mass,target_logit_vs_mean_other=margin)
        vectors[name]=vec
        summaries[name]={k:interval(v,weights) for k,v in vec.items()}
        summaries[name]['per_query']={str(q):{k:float(v[d['query_pair']==q].mean()) for k,v in vec.items()} for q in range(4)}
        summaries[name]['residual_mse_by_block_region']=d['residual_mse'].mean(0).tolist()
    contrasts={}
    def contrast(name,terms):
        contrasts[name]={metric:interval(sum(coef*vectors[condition][metric] for condition,coef in terms),weights) for metric in METRICS}
    for patch in ('guard','keypatch'):
        for graph in ('delete','add','transport'):
            contrast(f'{patch}_{graph}_minus_physical',[(patch+'_'+graph,1),(patch+'_physical',-1)])
        contrast(f'{patch}_add_effect_given_delete',[(patch+'_transport',1),(patch+'_delete',-1)])
        contrast(f'{patch}_delete_effect_given_add',[(patch+'_transport',1),(patch+'_add',-1)])
        contrast(f'{patch}_add_minus_delete',[(patch+'_add',1),(patch+'_delete',-1)])
        contrast(f'{patch}_edge_interaction',[(patch+'_transport',1),(patch+'_add',-1),(patch+'_delete',-1),(patch+'_physical',1)])
    for graph in ('physical','delete','add','transport'):
        contrast(f'keypatch_effect_{graph}',[('keypatch_'+graph,1),('guard_'+graph,-1)])
        contrast(f'native_minus_guard_{graph}',[('native',1),('guard_'+graph,-1)])
    contrast('state_graph_interaction',[('keypatch_transport',1),('guard_transport',-1),('keypatch_physical',-1),('guard_physical',1)])
    validity=json.loads((root/'validity.json').read_text())
    result={'phase':json.loads((root/'started.json').read_text())['phase'],'n':n,'validity':validity,
        'conditions':summaries,'contrasts':contrasts,'intervals':'Descriptive paired query-stratified percentile bootstrap;2000draws;seed10700003;no multiplicity-adjusted claims',
        'residual_region_order':['prefix','keys','values','query'],'equivalence_controls':['keypatch_transport','transport_all']}
    (root/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for name,vals in summaries.items():rows.append({'condition':name,**{k:vals[k]['mean'] for k in METRICS}})
    with (root/'conditions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# E7 exploratory state and graph decomposition','',f"Implementation validity: **{validity['all_pass']}**. {n} dictionaries, one GPT-2 checkpoint.",'',
        'Discovery data, not independent confirmation. Added edges and native residual patches are oracle diagnostics. Full restoration controls are guaranteed and do not count as discovery.','',
        '| Condition | 16-choice accuracy | Full-vocabulary accuracy | Raw target probability | Conditional target probability | Candidate mass | Target logit margin |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+row['condition']+' | '+' | '.join(f'{row[k]:.6f}' for k in METRICS)+' |')
    lines+=['','## Paired descriptive contrasts','','| Contrast | Restricted accuracy change | 95% interval | Conditional probability change | 95% interval |','| --- | ---: | --- | ---: | --- |']
    for name,d in contrasts.items():
        ac=d['restricted_accuracy'];pr=d['conditional_target_probability']
        lines.append(f"| {name} | {ac['mean']:.6f} | {ac['ci95']} | {pr['mean']:.6f} | {pr['ci95']} |")
    lines+=['','Every condition retains all inputs. Query-specific summaries, all six metrics and residual-distance profiles are in scores.json. These distances are descriptive and do not identify a mechanism.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'validity':validity['all_pass'],'n':n,'conditions':rows},indent=2))
if __name__=='__main__':main()
