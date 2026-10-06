"""Exploratory fixed-mask repair diagnostic; no new fit or neural forward."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
PILOT_SOURCE=HERE.parent/'world-model-planning-v1'
sys.path.insert(0,str(PILOT_SOURCE))
from world import probabilities,policy,episode

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 t0=time.monotonic();base=Path('outputs/world-model-planning/V1');out=Path('outputs/world-model-planning/REPAIR_V1');out.mkdir(exist_ok=False)
 pilot=json.loads((base/'plan.json').read_text());audit=json.loads((base/'INDEPENDENT_AUDIT.json').read_text())
 assert audit['status']=='PASS' and audit['report_sha256']==sha(base/'report.json')
 states=pilot['states'];table=np.array(pilot['transitions']);key=states.index([0,3,0])*4+2;door=states.index([1,2,1])*4+1
 masks={'key':[key],'door':[door],'both':sorted([key,door]),'all_heldout':pilot['test']}
 plan=dict(status='EXPLORATORY_POST_HOC_FIXED_MASK_DIAGNOSTIC',selected_after_pilot_results=True,model_fits=0,neural_forwards=0,provider_calls=0,new_spending_usd=0,masks=masks,horizon=16,cases=pilot['planning_cases'],
  inputs_sha256={str(base/n):sha(base/n) for n in ['plan.json','report.json','predictions.npz','INDEPENDENT_AUDIT.json']},
  source_sha256={str(p):sha(p) for p in [Path(__file__),PILOT_SOURCE/'world.py',PILOT_SOURCE/'audit.py']},
  intervention='Replace whole distribution on each named state/action row with exact one-hot true next state. Keep all other probabilities byte-identical. Same horizon16 planner and all600goal cases for all6fits.',
  rationale='Pilot energy models had better average transition accuracy but391/600H16successes; allmissed held-out doorentry. Test selected key/door and allheldout repairs without fitting or treating repair as deployable learning.',
  limits='This is a ground-truth intervention chosen after observing the pilot, not independent confirmation, a learned repair, an energy advantage or a unique explanation of natural learning.')
 save(out/'plan.json',plan)
 with np.load(base/'predictions.npz',allow_pickle=False) as z:base_p={n:probabilities(z[n]) for n in z.files}
 np.savez_compressed(out/'base_probabilities.npz',**base_p)
 repaired={};summaries={};episodes=[]
 for name,p in base_p.items():
  summaries[name]={}
  for variant,rows in masks.items():
   if time.monotonic()-t0>300:raise RuntimeError('Diagnostic timeout')
   matrix=p.copy()
   for flat in rows:
    s,a=divmod(flat,4);matrix[s,a,:]=0;matrix[s,a,table[s,a]]=1
   repaired[name+'__'+variant]=matrix
   policies={g:policy(matrix,states,g,16) for g in {tuple(c['goal']) for c in pilot['planning_cases']}}
   records=[]
   for case in pilot['planning_cases']:
    pi,ties=policies[tuple(case['goal'])];record=dict(model=name,repair=variant,horizon=16,**case,**episode(table,states,case['start'],case['goal'],pi,ties));records.append(record)
   episodes.extend(records)
   good=[r for r in records if r['success']]
   summaries[name][variant]=dict(n=600,successes=len(good),cycles=sum(r['cycle'] for r in records),mean_steps_success=float(np.mean([r['steps'] for r in good])) if good else None)
   print(json.dumps(dict(model=name,repair=variant,successes=len(good))),flush=True)
 np.savez_compressed(out/'repaired_probabilities.npz',**repaired)
 save(out/'episodes.json',episodes);save(out/'summary.json',dict(status='COLLECTED_PENDING_REPAIR_AUDIT',results=summaries,episodes=len(episodes),seconds=time.monotonic()-t0))
 save(out/'COLLECTED.json',dict(status='COLLECTED',files_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
if __name__=='__main__':main()
