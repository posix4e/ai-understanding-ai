"""Read-only independent E5 saved-outcome audit. No model/scorer imports."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path('outputs/experiment5'); RAW=ROOT/'confirmatory'
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ci(x,w):
 x=np.asarray(x,dtype=np.float64)
 return {'mean':float(np.mean(x)),'ci95':np.quantile(np.sum(w*x,axis=1),[.025,.975]).tolist()}
def weights(q,seed):
 rng=np.random.default_rng(seed); w=np.zeros((2000,len(q)))
 for s in range(4):
  ix=np.flatnonzero(q==s); w[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=2000)/len(q)
 return w
G=read('experiment5/conditions.json'); P=read('experiment5/protocol.json'); F=read('experiment5/predictions.json')['expected_aggregate']
with np.load(ROOT/'inputs.npz') as z: A={k:z[k] for k in z.files}
Q=A['query_pair']; T=A['targets']; VAL=A['values']; N=len(Q); BN=256
W=weights(Q,10200001); BW=weights(Q[:BN],10200002)
METRICS={'original_probability','accuracy','argmax_class','class_probability','l1_value_max_error_native','l1_query_max_error_native','l1_key_max_error_native','l1_key_max_error_baseline','l1_value_max_error_own_key_self','l2_input_value_max_error_native','l2_input_query_max_error_native','l2_input_key_max_error_native','class_probability_max_error_native'}
def audit(seed):
 ledger=read(RAW/f'seed_{seed}/index.json'); assert 'finished_at_utc' in ledger
 maxdiag={k:0. for k in METRICS if 'error' in k}; checked=0; shardcount=0; totalrows=0
 native=None; maxima={'noop':0.,'correct_value':0.,'correct_query':0.,'oracle':0.,'ownself':0.}
 def panel(name):
  nonlocal checked,shardcount,totalrows
  specs=G[name] if isinstance(G[name],list) else [G[name]]; spec={s['id']:s for s in specs}; found=set(); entries=ledger['panels'][name]['shards']; seenpaths=set()
  assert len(entries)==len(list((RAW/f'seed_{seed}'/name).glob('shard_*.npz')))
  for e in entries:
   p=Path(e['path']); assert str(p) not in seenpaths; seenpaths.add(str(p)); assert sha(p)==e['sha256']; shardcount+=1
   with np.load(p,allow_pickle=False) as z:
    d={}
    for k in z.files:
     l,c,m=k.split('/'); d.setdefault(l,{}).setdefault(c,{})[m]=z[k]
   assert set(d)==set(e['layout_ids'])
   for l,rr in d.items():
    assert l in spec and l not in found;found.add(l); layout=spec[l]; assert set(rr)=={c['id'] for c in layout['conditions']}
    n=BN if name=='boundary' else N
    for cid,m in rr.items():
     assert set(m)==METRICS
     for k,x in m.items(): assert x.shape==((n,16) if k=='class_probability' else (n,)) and np.isfinite(x).all()
     pp=m['class_probability']; yy=m['argmax_class']; assert ((pp>=0)&(pp<=1)).all(); np.testing.assert_allclose(pp.sum(1),1,rtol=1e-6,atol=1e-7)
     assert np.array_equal(pp.argmax(1),yy); assert np.array_equal(yy==T[:n],m['accuracy']);np.testing.assert_allclose(pp[np.arange(n),T[:n]],m['original_probability'],rtol=0,atol=1e-8)
     for k in maxdiag:
      assert (m[k]>=0).all();maxdiag[k]=max(maxdiag[k],float(m[k].max()))
     if native is not None: np.testing.assert_allclose(m['class_probability_max_error_native'],np.abs(pp-native[:n]).max(1),rtol=1e-5,atol=1e-7)
     checked+=1;totalrows+=n
    yield layout,rr
  assert found==set(spec)
 refs=list(panel('references')); ref=refs[0][1]; native=ref['native']['class_probability']; nativep=native[np.arange(N),T].astype(float); nativeacc=(native.argmax(1)==T)
 for m in ref.values(): np.testing.assert_allclose(m['class_probability_max_error_native'],np.abs(m['class_probability']-native).max(1),rtol=1e-5,atol=1e-7)
 correctP=[]; correctA=[]; controls=[]; cells=[]; best=[]; examples=[]; errlayouts={}; errrows=set(); grouped=None; grouped_acc=None; grouped_base=None
 for l,r in panel('primary'):
  c=r['correct']; cp=c['original_probability'].astype(float); ca=c['accuracy']; correctP.append(cp);correctA.append(ca.astype(float))
  maxima['noop']=max(maxima['noop'],float(np.abs(r['noop']['class_probability']-r['unguarded']['class_probability']).max())); maxima['oracle']=max(maxima['oracle'],float(np.abs(r['oracle']['class_probability']-native).max()))
  maxima['correct_value']=max(maxima['correct_value'],float(c['l1_value_max_error_native'].max()));maxima['correct_query']=max(maxima['correct_query'],float(c['l1_query_max_error_native'].max()))
  for q in range(4): cells.append({'layout':l['id'],'query':q,'accuracy':float(ca[Q==q].mean()),'probability':float(cp[Q==q].mean())})
  sh=[x['id'] for x in l['conditions'] if x['id'].startswith('sham_')]
  if sh:
   sp=np.stack([r[s]['original_probability'].astype(float) for s in sh]);controls.append(cp-np.mean(sp,axis=0)); bi=int(sp.mean(1).argmax());best.append({'layout':l['id'],'sham':sh[bi],'correct_minus_best':float(cp.mean()-sp[bi].mean()),'correct_probability':float(cp.mean()),'best_probability':float(sp[bi].mean()),'sham_count':len(sh)})
  wrong=np.flatnonzero(~ca.astype(bool));errlayouts[l['id']]=len(wrong);errrows.update(map(int,wrong))
  for row in wrong: examples.append({'layout':l['id'],'row':int(row),'query':int(Q[row]),'target':int(T[row]),'predicted':int(c['argmax_class'][row]),'target_probability':float(cp[row]),'native_probability':float(nativep[row]),'key_state_error':float(c['l1_key_max_error_native'][row])})
  if l['layout_indices']==[0,2,4,6,1,3,5,7,8]:grouped=c['class_probability'].astype(float);grouped_acc=float(ca[:BN].mean());grouped_base=float(r['unguarded']['accuracy'][:BN].mean())
 cp=np.mean(correctP,axis=0);ca=np.mean(correctA,axis=0);diff=np.mean(controls,axis=0);assert len(controls)==102 and len(cells)==420
 primary={'native_minus_correct':ci(nativep-cp,W),'correct_minus_mean_shams':ci(diff,W),'correct_probability':ci(cp,W),'correct_accuracy':ci(ca,W)}
 bsum={k:{'p':np.zeros(BN),'a':np.zeros(BN),'cells':[],'fixedp':np.zeros(BN),'fixeda':np.zeros(BN)} for k in ['unguarded','keyguard','logical_prefix','own_key_self']}; strata={};fixedcount=0
 for l,r in panel('boundary'):
  maxima['ownself']=max(maxima['ownself'],float(r['own_key_self']['l1_value_max_error_own_key_self'].max()));fixed=l['layout_indices'][:4]==[0,2,4,6];fixedcount+=fixed
  for k,b in bsum.items():
   b['p']+=r[k]['original_probability'];b['a']+=r[k]['accuracy']
   if fixed:b['fixedp']+=r[k]['original_probability'];b['fixeda']+=r[k]['accuracy']
   for q in range(4):
    a=float(r[k]['accuracy'][Q[:BN]==q].mean());missing=l['missing_native_predecessors'][q];mk=sum(x%2==0 for x in missing);mv=sum(x%2==1 for x in missing)
    b['cells'].append({'layout':l['id'],'query':q,'accuracy':a,'missing_keys':mk,'missing_values':mv});ss=strata.setdefault(k,{}).setdefault(f'keys{mk}_values{mv}',[0.,0]);ss[0]+=a;ss[1]+=1
 boundary={}
 for k,b in bsum.items():
  boundary[k]={'probability':ci(b['p']/2415,BW),'accuracy':ci(b['a']/2415,BW),'passing_cells':sum(x['accuracy']>=.95 for x in b['cells']),'total_cells':9660,'min_accuracy':min(x['accuracy'] for x in b['cells']),'worst_cells':sorted(b['cells'],key=lambda x:x['accuracy'])[:10],'fixed_key_order_accuracy':ci(b['fixeda']/fixedcount,BW)}
 assert fixedcount==23
 edge_means=[];edge_draws=[];peredge={}
 for l,r in panel('edge_panel'):
  for c in l['conditions']:
   i,j=c['edge'];rows=np.flatnonzero(Q==j); wrong=[k for k in range(4) if k!=i and k!=j];delta=r[c['id']]['class_probability'].astype(float)-grouped;delta=delta[np.arange(N)[:,None],VAL];x=delta[rows,i]-delta[rows][:,wrong].mean(1);w=W[:,rows]*4;edge_means.append(float(x.mean()));edge_draws.append(np.sum(w*x,axis=1));peredge[c['id']]=ci(x,w)
 edge={'mean':float(np.mean(edge_means)),'ci95':np.quantile(np.mean(edge_draws,axis=0),[.025,.975]).tolist()}
 aggregate={'native_mean_probability':float(nativep.mean()),'native_accuracy':float(nativeacc.mean()),'primary_correct_mean_probability':float(cp.mean()),'primary_correct_accuracy':float(ca.mean()),'primary_correct_min_layout_query_accuracy':min(c['accuracy'] for c in cells),'primary_native_minus_correct_probability':primary['native_minus_correct']['mean'],'primary_correct_minus_mean_shams_probability':primary['correct_minus_mean_shams']['mean'],'edge_wrong_value_difference_of_changes':edge['mean']}
 for k in ['logical_prefix','own_key_self']:aggregate[f'boundary_{k}_accuracy']=boundary[k]['accuracy']['mean'];aggregate[f'boundary_{k}_fraction_layout_query_accuracy_ge_0_95']=boundary[k]['passing_cells']/9660
 assert set(aggregate)==set(F[str(seed)])
 forecast=[{'endpoint':k,'forecast':float(F[str(seed)][k]),'observed':v,'absolute_error':abs(v-F[str(seed)][k]),'within_0_05':abs(v-F[str(seed)][k])<=.05} for k,v in aggregate.items()]
 valid=aggregate['native_accuracy']>=.95 and maxima['noop']<=1e-6 and all(maxima[k]<=1e-5 for k in ['correct_value','correct_query','oracle','ownself']) and maxdiag['l1_query_max_error_native']<=1e-5 and maxdiag['l1_key_max_error_baseline']<=1e-5
 gates={'valid':bool(valid),'every_layout_query':aggregate['primary_correct_min_layout_query_accuracy']>=.95,'native_deficit':primary['native_minus_correct']['ci95'][1]<=.01,'mean_sham_specificity':primary['correct_minus_mean_shams']['ci95'][0]>.02,'boundary_logical_prefix':boundary['logical_prefix']['accuracy']['mean']>=.95,'boundary_own_key_self':boundary['own_key_self']['accuracy']['mean']>=.95,'edge_enrichment':edge['ci95'][0]>.05}
 return {'seed':seed,'checked_cells':checked,'checked_condition_rows':totalrows,'verified_shards':shardcount,'aggregate':aggregate,'primary':primary,'gates':gates,'manipulation_maxima':maxima,'all_metric_maxima':maxdiag,'failed_primary_cells':[x for x in cells if x['accuracy']<.95],'primary_error_instances':len(examples),'primary_error_unique_rows':len(errrows),'primary_layouts_with_errors':sum(x>0 for x in errlayouts.values()),'worst_primary_examples':sorted(examples,key=lambda x:x['target_probability'])[:20],'best_shams_beating_correct_point':sum(x['correct_minus_best']<0 for x in best),'closest_or_better_shams':sorted(best,key=lambda x:x['correct_minus_best'])[:10],'boundary':boundary,'boundary_missing_context_strata':{p:{k:{'mean_accuracy':v[0]/v[1],'cells':v[1]} for k,v in ss.items()} for p,ss in strata.items()},'grouped_reference_first256_accuracy':{'unguarded':grouped_base,'correct':grouped_acc},'edge':edge,'per_edge':peredge,'forecasts':forecast}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);parser.add_argument('--final',action='store_true');args=parser.parse_args()
 if args.seed is not None:
  result=audit(args.seed);save(f'work/e5_audit_seed_{args.seed}.json',result);print(json.dumps({k:result[k] for k in ['seed','checked_cells','gates','aggregate','primary_error_instances']}));return
 lock=read(ROOT/'remote_lock.json');bad=[p for p,h in lock['files'].items() if sha(p)!=h];assert not bad
 start=read(RAW/'start.json');assert start['preregistration_commit']==lock['commit'];assert datetime.fromisoformat(lock['verified_at_utc'])<datetime.fromisoformat(start['utc'])
 out={'hashes_verified':len(lock['files']),'frozen_commit':lock['commit'],'verified_at_utc':lock['verified_at_utc'],'start_utc':start['utc'],'model_forwards_run_by_auditor':False}
 if args.final:
  ss={str(s):read(f'work/e5_audit_seed_{s}.json') for s in range(6,12)};official=read(RAW/'scores.json');deltas=[]
  for s,r in ss.items():
   for k,v in r['aggregate'].items():deltas.append(abs(v-official[s]['aggregate'][k]))
   for k,v in r['primary'].items():deltas.extend(abs(np.asarray([v['mean'],*v['ci95']])-np.asarray([official[s]['comparisons'][k]['mean'],*official[s]['comparisons'][k]['ci95']])))
   for k,v in r['boundary'].items():
    for m in ['probability','accuracy']:deltas.extend(abs(np.asarray([v[m]['mean'],*v[m]['ci95']])-np.asarray([official[s]['boundary'][k][m]['mean'],*official[s]['boundary'][k][m]['ci95']])))
   deltas.extend(abs(np.asarray([r['edge']['mean'],*r['edge']['ci95']])-np.asarray([official[s]['edge']['mean'],*official[s]['edge']['ci95']])))
  assert max(deltas)<1e-6
  out.update({'finish_utc':read(RAW/'finish.json')['utc'],'independent_official_max_absolute_difference':float(max(deltas)),'total_checked_cells':sum(r['checked_cells'] for r in ss.values()),'total_checked_condition_rows':sum(r['checked_condition_rows'] for r in ss.values()),'seeds':ss,'forecast_within_0_05':sum(f['within_0_05'] for r in ss.values() for f in r['forecasts']),'forecast_total':72})
  save(ROOT/'independent_audit.json',out)
 print(json.dumps({k:v for k,v in out.items() if k!='seeds'},indent=2))
if __name__=='__main__':main()
